# Testing Strategy: End-to-End Audio Pipeline

## Current state

Unit tests cover post-processing stages (ITN, spelling, grammar, filler removal) and
utility code (VAD prefill buffer, speaker cosine similarity, model registry). All run
with `just test` in under 15 seconds.

No tests currently exercise the audio-to-text path. Bugs in VAD boundaries, chunk
joining, and short utterance handling have been caught only through manual testing on
device. This document outlines a layered test plan to close that gap.

## Layer 1: VAD boundary tests (unit, no model needed)

Cheapest to write and fastest to run. Use synthetic audio generated in code.

### What to test

- **Prefill prepended on flush.** Feed silence then a sine burst, stop mid-speech,
  call flush(). Assert returned segment length equals prefill + speech frames.
- **Prefill frozen during speech.** Feed silence, then speech. Assert prefill buffer
  stops growing once detector.detected() is true.
- **No duplication.** Feed a short burst (< 500ms), flush. Assert the speech portion
  appears exactly once in the output (compare sample values, not just length).
- **Prefill reset after segment.** Feed speech then silence until a complete segment
  is emitted via accept(). Feed more silence then speech. Assert the second segment's
  prefill contains only audio from after the first segment completed.
- **Force-split boundary.** Feed continuous speech exceeding MAX_SEGMENT_SECS. Assert
  the split produces two non-overlapping segments.
- **Audio drain on stop.** Simulate buffered audio in the channel at stop time. Assert
  all buffered samples are included in the final output.

### Implementation notes

- Generate audio with `fn sine_burst(freq: f32, duration_ms: u32, sample_rate: i32) -> Vec<f32>`
- Silence is just `vec![0.0; n]`
- These tests need the Silero VAD ONNX file. Use `ModelManager::ensure_vad_model()` in
  test setup, or gate behind `#[cfg(feature = "integration")]` if you want `cargo test`
  to stay fast without model files.
- Alternatively, test the PrefillBuffer and audio-drain logic in isolation (no VAD model
  needed). The prefill freeze behavior requires the real VAD to know when speech starts.

## Layer 2: Transcription accuracy tests (integration, needs ONNX model)

Feed known audio clips through the transcriber and assert on output text.

### Reference clip set

Start with clips that cover known failure modes:

| Clip | Duration | Expected output | Tests |
|------|----------|----------------|-------|
| single-word-hello.wav | ~0.5s | "hello" | Short utterance recognition |
| single-word-stop.wav | ~0.4s | "stop" | Trailing stop consonant |
| single-word-yes.wav | ~0.3s | "yes" | Very short utterance |
| short-phrase.wav | ~1.5s | "turn it off" | 3-word phrase boundary |
| counting.wav | ~3s | "1, 2, 3, 4, 5" | ITN + comma separation |
| long-sentence.wav | ~8s | (full sentence) | Normal-length baseline |
| trailing-word.wav | ~1s | "testing one two" | Last word not clipped |
| soft-spoken.wav | ~1s | "hello" | Low volume / VAD threshold |

### Assertion strategy

- Exact match for clean, controlled recordings
- For noisier clips or where minor variation is acceptable, use word error rate (WER)
  with a threshold (e.g., WER < 0.1)
- A simple WER function: `fn wer(expected: &str, actual: &str) -> f32` that computes
  edit distance on word arrays divided by expected word count

### Recording the clips

- Record on the target device (Android) in a quiet room
- Save as 16kHz mono WAV (the format the transcriber expects)
- Keep clips small. The full set should be under 1 MB.

## Layer 3: Full pipeline tests (integration)

Feed audio through VAD -> transcriber -> post-processing and assert on final text.
These cover the entire chain including chunk joining, filler removal, ITN, spelling,
and grammar correction.

### What to test

- **Chunk joining.** Two VAD segments from one recording produce coherent joined text
  without false sentence boundaries or dropped words.
- **Post-processing preserves meaning.** Spelling correction and grammar don't mangle
  domain-specific words or short phrases.
- **Filler removal.** Audio with "um" and "uh" produces clean output.
- **ITN in context.** "I paid twenty three dollars" becomes "I paid $23" end to end.

### Implementation

```rust
fn pipeline_test(audio: Vec<f32>, expected: &str) {
    let transcriber = Transcriber::new(...);
    let raw = transcriber.transcribe(audio, 16_000).unwrap();
    let result = postprocess::postprocess(&raw);
    assert_eq!(result.text, expected);
}
```

## Where to store test fixtures

- `src-tauri/tests/fixtures/` for audio clips (WAV files)
- Add to `.gitignore` if clips are large; download from a known URL in CI instead
- For clips under 1 MB total, committing to the repo is fine

## How to run

Gate integration tests so `just test` (unit tests) stays fast:

```rust
#[cfg(feature = "integration")]
#[test]
fn transcribes_single_word() {
    let audio = load_wav("tests/fixtures/single-word-hello.wav");
    let transcriber = setup_transcriber();
    let result = transcriber.transcribe(audio, 16_000).unwrap();
    assert_eq!(result.trim(), "hello");
}
```

Run with: `cargo test --features integration --lib`

Or use `#[ignore]` and run with: `cargo test --lib -- --ignored`

## Priority order

1. **VAD boundary unit tests** (PrefillBuffer isolation + synthetic audio with real VAD).
   Catches duplication bugs, clipping, prefill freeze behavior. Fast, no audio fixtures.
2. **Single-word transcription clips.** The failure mode that prompted this document.
   Requires model files and recorded clips but directly validates the fix.
3. **Full pipeline tests.** Highest coverage but slowest to set up and most brittle
   (model updates change output). Add incrementally as bugs surface.

## Regression workflow

When a transcription bug is found:
1. Record or isolate the audio that triggers it
2. Add it to the fixture set with the expected output
3. Fix the bug
4. The clip stays as a regression test permanently
