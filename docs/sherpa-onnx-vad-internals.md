# sherpa-onnx VAD Internals

Documents the internal behaviour of `VoiceActivityDetector` from sherpa-onnx
that directly affects how we handle audio in `vad.rs`. Read this before
touching the VAD or adding any kind of pre-speech buffering.

## Built-in onset lookback

When speech is first detected, sherpa-onnx **backdates the segment start**
by `2 * window_size + min_speech_duration_samples` relative to the current
buffer tail:

```cpp
// voice-activity-detector.cc, AcceptWaveform()
start_ = std::max(
    buffer_.Tail() - 2 * model_->WindowSize() - model_->MinSpeechDurationSamples(),
    buffer_.Head()
);
```

With our configured parameters:
- `window_size` = 512 samples (32ms at 16kHz)
- `min_speech_duration` = 0.1s = 1600 samples

Lookback = 2×512 + 1600 = **2624 samples ≈ 164ms**

For the very first utterance in a recording the lookback clamps to
`buffer_.Head()`, so it goes all the way back to the start of available
audio. No word clipping occurs even for the first word.

## `detected()` semantics

`IsSpeechDetected()` (what our `detected()` maps to) returns `start_ != -1`.
`start_` is set on the **first frame** where the Silero model scores above
threshold. There is no `min_speech_duration` confirmation delay — that
parameter only controls the minimum length of completed segments, not the
onset detection timing.

## Window processing granularity

The VAD processes audio in 512-sample (32ms) windows. Our recorder sends
256-sample (16ms) chunks. The internal `last_` accumulator collects incoming
samples until a full window is available:

```cpp
last_.insert(last_.end(), samples, samples + n);
if (last_.size() < window_size) {
    return;  // wait for more samples
}
```

This means `detected()` can only transition to `true` after at least 2
consecutive 16ms chunks have been received.

## Do NOT add an external prefill

It is tempting to prepend a ring buffer of pre-speech audio to each segment
to "help" the transcription model hear the word onset. **Do not do this.**

The problem: audio arrives in 16ms chunks, but the VAD needs 32ms to make a
decision. The first 16ms chunk of speech (call it chunk A) is fed to the
internal `last_` accumulator. At that moment, `detected()` is still false
(the window isn't complete yet). If you push chunk A to a prefill buffer at
this point, and the next 16ms chunk (B) completes the window and triggers
detection — chunk A now appears in **both** the prefill and the sherpa-onnx
segment's built-in lookback. The transcription model hears the first 16ms of
speech twice, which manifests as onset word repetition:

> "What whatever you think" instead of "Whatever you think"

The sherpa-onnx 164ms lookback is sufficient for all practical onset cases.
Trust it and don't layer anything on top.

## Silence trimming between segments

When `start_ == -1` (no active speech), the buffer continuously pops old
samples to retain only the lookback window:

```cpp
int32_t end = buffer_.Tail() - 2 * model_->WindowSize() -
              model_->MinSpeechDurationSamples();
int32_t n = std::max(0, end - buffer_.Head());
if (n > 0) buffer_.Pop(n);
```

This keeps memory bounded during long silences. It also means the 164ms
lookback at the next speech onset comes from this retained silence, not from
scratch.

## Segment end trimming

When silence is detected after speech, the segment end is backdated by
`min_silence_duration_samples` to trim the trailing silence:

```cpp
int32_t end = buffer_.Tail() - model_->MinSilenceDurationSamples();
std::vector<float> s = buffer_.Get(start_, end - start_);
```

With `min_silence_duration = 0.3s`, segments are trimmed by 300ms at the
end. This is intentional — the silence that triggered end-of-speech
detection is not part of the speech content.
