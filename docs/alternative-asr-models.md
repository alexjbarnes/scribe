# Alternative ASR Models

Models available through sherpa-onnx that could be added to Scribe. All have
pre-converted ONNX files on HuggingFace. Organized by relevance to our use case.

## Moonshine (Useful Sensors)

Encoder-decoder transformer designed for on-device use. Variable-length input
(no 30-second padding like Whisper). Uses RoPE instead of absolute position
embeddings.

### Variants

**V1 (Oct 2024) - English only, MIT license:**

| Model | Params | INT8 Size | WER (LS clean) | WER (LS other) |
|-------|--------|-----------|---------------|---------------|
| Tiny | 27M | ~118 MB | 4.52% | 11.71% |
| Base | 62M | ~273 MB | 3.23% | 8.18% |

V1 Tiny beats Whisper tiny.en on all benchmarks despite fewer parameters.
V1 Base beats Whisper base.en across the board.

**V2 "Flavors of Moonshine" (Sep 2025) - Per-language models:**
Same tiny/base sizes, individually trained for: Arabic, Chinese, Japanese,
Korean, Ukrainian, Vietnamese, Spanish. Claims 48% lower error than Whisper
Tiny on average across languages. License: "other" (permissive).

**Streaming (Jan 2026) - English only, MIT license:**

| Model | Params | RPi5 Latency | WER (LS clean) | WER (LS other) |
|-------|--------|-------------|---------------|---------------|
| Tiny | 34M | 237ms | 4.49% | 12.09% |
| Small | 123M | 527ms | 2.49% | 6.78% |
| Medium | 245M | 802ms | 2.08% | 5.00% |

Streaming Medium beats Whisper Large v3 in both accuracy and speed on RPi5.
No sherpa-onnx ONNX conversions exist yet for streaming models.

### Speed comparison (RPi5, 10s audio)

| Model | Latency | Relative |
|-------|---------|----------|
| Moonshine Tiny Streaming | 237ms | 1x |
| Moonshine Small Streaming | 527ms | 2.2x |
| Moonshine Medium Streaming | 802ms | 3.4x |
| Whisper Tiny | 5,863ms | 24.7x |
| Whisper Small | 10,397ms | 43.9x |

### sherpa-onnx integration

Dedicated engine: `OfflineMoonshineModelConfig`. Not whisper, not transducer,
not CTC. Requires 4-5 ONNX files per model (preprocess, encode, cached_decode,
uncached_decode, tokens).

Pre-converted INT8 repos:
- `csukuangfj/sherpa-onnx-moonshine-tiny-en-int8`
- `csukuangfj/sherpa-onnx-moonshine-base-en-int8`
- V2 per-language repos under `csukuangfj2/`

### Assessment

Strong candidate for desktop lightweight option. Tiny at 118 MB INT8 offers
Whisper-base-level accuracy at a fraction of the size and dramatically faster
on ARM. On mobile, still an encoder-decoder (autoregressive decode), so
Parakeet's transducer architecture will be faster per inference. But Moonshine's
much smaller model size means faster loading and lower memory. Worth testing
on Android to see how the speed/accuracy compares to Parakeet in practice.

---

## Canary (NVIDIA NeMo)

Encoder-decoder using FastConformer encoder (same as Parakeet) with autoregressive
Transformer decoder. Same NeMo team as Parakeet. CC-BY-4.0 license.

Key difference from Parakeet: encoder-decoder (like Whisper) vs transducer.
Supports translation natively. Slower inference than Parakeet due to
autoregressive decoding.

### Variants

| Model | Params | Languages | INT8 Size | Release |
|-------|--------|-----------|-----------|---------|
| 180M Flash | 182M | en/de/es/fr + translation | ~207 MB | Mar 2025 |
| 1B Flash | 883M | en/de/es/fr + translation | n/a | Mar 2025 |
| 1B v2 | 978M | 25 European + translation | n/a | Aug 2025 |
| Qwen 2.5B | 2.5B | English only (LLM post-proc) | n/a | Jun 2025 |

### English accuracy (Open ASR Leaderboard mean WER)

| Model | Mean WER | RTFx |
|-------|---------|------|
| Parakeet TDT 0.6B v2 | 6.05% | 3380x |
| Canary 1B Flash | 6.35% | 1046x |
| Canary 180M Flash | ~7.07% | 1233x |
| Canary 1B v2 | 7.15% | 749x |
| Canary Qwen 2.5B | 5.63% | 418x |

Parakeet beats all Canary models on English except Qwen 2.5B (which is 4x larger
and 8x slower).

### sherpa-onnx integration

Dedicated engine: `offline_canary`. Requires encoder + decoder + tokens.

Only 180M Flash has sherpa-onnx ONNX conversions:
- `csukuangfj/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8` (207 MB)

No conversions exist for 1B Flash, 1B v2, or Qwen 2.5B.

### Assessment

Canary 180M Flash INT8 at 207 MB is the interesting one: small, multilingual (4 langs),
translation support. Could be the multilingual mobile option instead of Whisper. English
accuracy is worse than Parakeet (~7% vs ~6% WER) but much better than Whisper at similar
size. The autoregressive decoder means it will be slower than Parakeet on mobile, same
architectural penalty as Whisper. Best suited for desktop multilingual use.

---

## SenseVoice (Alibaba / FunAudioLLM)

Non-autoregressive encoder-only model (Small variant). Single forward pass, no
sequential decoding. 5x faster than Whisper-Small, 15x faster than Whisper-Large.

### Variants

| Model | Params | Languages | INT8 Size |
|-------|--------|-----------|-----------|
| Small | 234M | 50+ (best: zh/en/ja/ko/yue) | ~228 MB |
| Large | 1,587M | 50+ | n/a |

### Special features

- Speech emotion recognition (happy, sad, angry, neutral, fearful, disgusted, surprised)
- Audio event detection (BGM, applause, laughter, crying, coughing, sneezing)
- Spoken language identification
- Built-in inverse text normalization

### English accuracy (Small)

| Dataset | SenseVoice-Small | Whisper-Small |
|---------|-----------------|--------------|
| LS test-clean | 3.15% | 3.13% |
| LS test-other | 7.18% | 7.37% |
| AISHELL-1 (Mandarin) | 2.96% | 10.04% |

Roughly matches Whisper-Small on English. Destroys Whisper on Mandarin/CJK.

### sherpa-onnx integration

Dedicated engine: `OfflineSenseVoiceModelConfig`. Single ONNX file + tokens.
Config has `language` field ("auto", "en", "zh", etc.) and `use_itn` flag.

Pre-converted:
- `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` (INT8: 228 MB)

### License

FunASR Model Open Source License v1.1. Commercial use permitted with attribution.
Has a clause that revokes the license if you publicly denigrate the project. Not
MIT/Apache.

### Assessment

Best option for CJK language support. Non-autoregressive architecture means it
should be fast on mobile (single forward pass, no sequential decoding like Whisper
or Canary). The license clause about public denigration is unusual and worth noting.
English accuracy matches Whisper-Small but trails Parakeet significantly (~7% vs ~6%
mean WER). Primary value is for Chinese/Japanese/Korean/Cantonese users.

---

## GigaAM v3 (Sberbank / Salute Developers)

Conformer-based model for Russian. Pre-trained on 700K hours of Russian speech.
MIT license.

### Variants

| Model | Type | Mean WER (Russian) |
|-------|------|-------------------|
| CTC | CTC decoder | 9.2% |
| RNNT | RNN-Transducer | 8.4% |
| E2E CTC | CTC + punctuation/ITN | 9.2% |
| E2E RNNT | RNNT + punctuation/ITN | 8.4% |

Claims 50% better than Whisper-large-v3 on Russian.

### sherpa-onnx integration

CTC variant: `nemo_ctc` engine. RNNT variant: `transducer` engine.

Pre-converted (float32 only, no INT8):
- `sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16`
- `sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16`

### Assessment

Only relevant for Russian language support. MIT license is clean. The RNNT variant
uses the same transducer engine as Parakeet, so integration would be straightforward.
No INT8 quantized versions available yet.

---

## Breeze ASR

Optimized for Taiwanese Mandarin with code-switching support. 1.1 GB download.
Niche use case. No further research done.

---

## Summary: What to consider adding

**High priority (clear value):**
- Moonshine Tiny INT8 (118 MB) - lightweight English option for desktop
- Moonshine Base INT8 (273 MB) - better accuracy, still small
- Canary 180M Flash INT8 (207 MB) - compact multilingual (4 European langs) + translation

**Medium priority (specific language needs):**
- SenseVoice INT8 (228 MB) - best CJK support, non-autoregressive speed
- GigaAM v3 (~152 MB) - best Russian support

**Low priority (wait for sherpa-onnx conversions):**
- Moonshine Streaming models - would be great for mobile but no ONNX conversions yet
- Canary 1B v2 - 25 European languages but no sherpa-onnx conversion
- Canary Qwen 2.5B - best English accuracy overall but huge and slow

**Each new model family requires a new engine type in the codebase.** Currently
we support `whisper` and `parakeet` (transducer). Adding Moonshine, Canary, or
SenseVoice each requires mapping to their respective sherpa-onnx config structs.
GigaAM's RNNT variant would work through the existing transducer engine.
