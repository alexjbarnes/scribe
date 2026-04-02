# Hotword / Context Biasing in sherpa-onnx

Hotwords (also called context biasing) let you boost the probability of specific words or phrases during decoding. Useful for proper nouns, brand names, and technical terms the base model consistently misrecognises.

## Which models support hotwords

| Model type | Hotwords supported |
|---|---|
| Offline Transducer (Parakeet TDT) | Yes — requires `modified_beam_search` |
| Offline NeMo CTC (Conformer CTC) | No |
| Offline Whisper | No |
| Online Transducer | Yes |
| Online CTC | No |

**Only transducer models support hotwords.** The sherpa-onnx Rust API exposes `hotwords_file` on `OfflineRecognizerConfig` for all model types, but the underlying C++ silently ignores it for CTC and Whisper. This is documented officially at https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html.

## Parakeet TDT hotword support

Support was added in sherpa-onnx v1.12.24 via PR #3077 (merged Feb 5, 2026). The project is on v1.12.34 so it is available.

PR #3077 added `OfflineTransducerModifiedBeamSearchNeMoDecoder`, which handles TDT-specific quirks:
- Correct blank token position (`vocab_size - 1`, not `0`)
- Proper LSTM state cloning per beam hypothesis
- TDT frame-offset tracking during beam search

### Requirements

1. `decoding_method = "modified_beam_search"` — greedy search does not run the hotword scoring path
2. `hotwords_file` pointing to a prepared hotwords file
3. `hotwords_score` — boost weight (start around `1.5`, tune up/down)
4. `bpe.vocab` file present alongside the model files
5. `modeling_unit = "bpe"` in model config

### Hotwords file format

Plain text, one phrase per line:

```
FaceTime
GitHub Actions
ARM64
Starbucks
```

Before use, convert to token-level representation using the `text2token.py` script from the sherpa-onnx repo:

```bash
python3 text2token.py \
  --tokens /path/to/tokens.txt \
  --words hotwords_plain.txt \
  > hotwords_tokens.txt
```

Optionally append a per-phrase score override:

```
FaceTime :3.5
GitHub Actions :2.0
```

### Wiring it up in Rust

In `transcribe.rs`, `create_recognizer()`, add to the `Transducer` branch:

```rust
config.decoding_method = Some("modified_beam_search".into());
config.hotwords_file = Some("/path/to/hotwords_tokens.txt".into());
config.hotwords_score = 1.5;
```

Or use per-stream hotwords instead of a file (hotwords already tokenised, `/`-separated between phrases):

```rust
let stream = recognizer.create_stream_with_hotwords("▁Face Time / ▁Git Hub ▁Actions");
```

## Known limitations

- **WER degradation**: Multiple users report that enabling hotwords increases the word error rate on general speech. The boost affects beam pruning globally. Keep `hotwords_score` conservative (1.5–2.0) and test carefully.
- **CTC has no equivalent**: For Conformer CTC or Whisper, the only option is post-processing substitution rules (e.g. map "maine" → "Maine" when it follows "state of").
- **Token preparation is a manual step**: The `text2token.py` script is not bundled with the app; hotword files need to be pre-processed at build time, not at runtime.

## Alternatives for CTC / Whisper

Since hotwords are not available for those model types, consider:

1. **Post-processing substitution** — add rules to the `itn` or a new `vocab` stage for known misrecognitions specific to your user base.
2. **Fine-tuning** — adapt the base model on domain-specific data. Not practical for a general-purpose app.
3. **Switch to Parakeet TDT** — if hotword biasing matters, Parakeet TDT is the only model in the supported set that enables it.
