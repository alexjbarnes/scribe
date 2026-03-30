# Speaker Diarization

Researched March 2026. Speaker diarization detects and labels different speakers in audio, producing timestamped segments with speaker IDs.

## sherpa-onnx has it built in

The `sherpa-onnx = "1.12"` crate already in Cargo.toml exposes a complete offline speaker diarization API. No new dependency needed.

Rust API surface:
- `OfflineSpeakerDiarization::create(config) -> Option<Self>`
- `OfflineSpeakerDiarization::process(samples: &[f32]) -> Option<OfflineSpeakerDiarizationResult>`
- Result contains `OfflineSpeakerDiarizationSegment` entries with `start`, `end`, and `speaker` fields

Separate `SpeakerEmbeddingExtractor` and `SpeakerEmbeddingManager` APIs exist for speaker identification (registering known voices and matching against them). Different use case from diarization but uses the same embedding models.

## How it works

Three-stage pipeline:

1. **Segmentation**: A neural network (pyannote-style) processes overlapping 10-second windows with a sliding window. For each frame, predicts which of up to 3 speakers is active (including overlap). Speaker IDs are only locally consistent within each window.

2. **Embedding extraction**: For each detected speaker segment, an ECAPA-TDNN or TitaNet model extracts a fixed-dimensional vector (192 or 512 dims) encoding voice characteristics.

3. **Clustering**: Agglomerative clustering groups embeddings across windows to produce globally consistent speaker labels. A threshold parameter controls granularity (larger = fewer speakers). You can also specify `num_clusters` directly if you know the speaker count.

## Models required

Two models: segmentation + embedding.

### Segmentation models (~6MB each)

| Model | Size | Notes |
|-------|------|-------|
| pyannote-segmentation-3.0 | ~6MB | Community ONNX export, ~2.2M params |
| reverb-diarization-v1 | ~6MB | Fine-tuned by Rev.com on transcription data |

Both output a (frames, 7) matrix per 10-second window: non-speech, speaker 1-3, and overlap combinations.

Available at: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models

### Embedding models (~25MB each)

| Model | Size | Params | Notes |
|-------|------|--------|-------|
| WeSpeaker ECAPA-TDNN 512 | ~25MB | ~6M | VoxCeleb2 trained |
| NeMo TitaNet-Small | ~25MB | 6.4M | English speaker verification |
| NeMo TitaNet-Large | ~95MB | 25.3M | Higher accuracy, too large for mobile |
| 3D-Speaker ERes2Net | ~25-30MB | ~6M | Mandarin-focused |

INT8 quantized variants available, roughly halving sizes.

Available at: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models

**Total model footprint**: ~30-50MB (segmentation + embedding). ~15-25MB with INT8.

## Performance

- Processing a 60s recording: 1-3 seconds on a phone CPU
- Segmentation runs at 50-200x real-time
- Embedding extraction: fast per segment
- Clustering: negligible for small speaker counts

Accuracy: DER (Diarization Error Rate) ~11-19% on standard benchmarks for 2-3 speakers. Degrades with more speakers or heavy overlap.

Language-independent: works on voice characteristics, not language content.

## Constraint: offline only

`OfflineSpeakerDiarization` processes a complete audio buffer, not a stream. No real-time speaker labels during recording. Must wait until recording finishes, then process the full audio.

## Primary approach: speaker verification

Speaker verification (not full diarization) is the right primitive for both use cases. Uses `SpeakerEmbeddingExtractor` from sherpa-onnx to extract a voice fingerprint per VAD segment, then compares against a registered speaker embedding via cosine similarity.

### How it works

1. **Enrollment (one-time):** User records a few seconds of speech. Extract embedding, store it.
2. **Per-segment check:** After VAD fires a segment, extract its embedding (~10-50ms on phone CPU). Compare against stored embedding via cosine similarity (microseconds).
3. **Decision:** Above threshold = target speaker, below = discard or label differently.

### Latency impact

- Embedding extraction: ~10-50ms per VAD segment (ECAPA-TDNN, ~6M params)
- Cosine similarity: microseconds
- Runs in parallel with transcription of the previous segment, so effectively zero added latency on the critical path
- Embedding model: ~25MB (~12MB INT8), embedded in binary

### Mobile: speaker filtering

When dictating, background voices (TV, other people) contaminate the transcript. Speaker verification discards non-matching segments before they reach the transcriber.

Flow:
```
VAD fires segment
  -> Extract speaker embedding (~10-50ms)
  -> Cosine similarity against enrolled voice
  -> Below threshold? Discard segment, skip transcription
  -> Above threshold? Send to transcriber as normal
```

### Desktop: meeting transcription

Same embedding extraction per segment, but instead of discard/keep, tag each segment with a speaker label. Two modes:

**Known speakers (enrolled):** Match each segment against enrolled embeddings. Label with the closest match above threshold, or "Unknown" if none match.

**Unknown speakers (clustering):** Collect all segment embeddings, cluster them (agglomerative), assign speaker IDs. This is what `OfflineSpeakerDiarization` does as a post-processing step after recording stops.

Output format:
```
[Speaker 1] Hey, did you finish the report?
[Speaker 2] Almost. I need the Q3 numbers from you.
[Speaker 1] Sending them now.
```

Latency is less critical for meetings. A few seconds of post-processing for a 60s recording is acceptable.

## Integration considerations

- `SpeakerEmbeddingExtractor` is the core building block for both mobile and desktop use cases
- Embedding model (~12MB INT8) embedded in binary via `include_bytes!`
- Per-segment embedding extraction can overlap with transcription (pipeline parallelism)
- Enrollment data stored as a small float vector in app config/data directory
- Cosine similarity threshold needs tuning (typical range: 0.5-0.7)
- On Android, consider running in the forked process alongside transcription

## Full diarization API

For meeting mode without enrolled speakers, sherpa-onnx also provides `OfflineSpeakerDiarization`:
- Requires segmentation model (pyannote 3.0, ~6MB) in addition to embedding model
- Processes complete audio buffer (offline only, not streaming)
- Uses sliding window segmentation + clustering for globally consistent speaker labels
- `OfflineSpeakerDiarization` is `Send` but not `Sync` (single-thread access)

## Alternatives considered

| Option | Verdict |
|--------|---------|
| Picovoice Falcon | Proprietary, requires API key. Not offline-first |
| EEND (end-to-end neural) | Research-stage, no production ONNX models |
| Voxtral Transcribe 2 | Joint ASR+diarization but multi-GB LLM. Not mobile-viable |
| Custom pipeline | sherpa-onnx already does this, no reason to rebuild |
