# Model Research

Evaluation of ASR models for potential inclusion. Researched and updated through April 2026.

---

## Viable candidates (sherpa-onnx, not yet added)

### Distil-Whisper (HuggingFace / Hugging Face + OpenAI)

Distilled Whisper variants with identical architecture to the original — same encoder/decoder format, fewer decoder layers. Uses the existing Whisper engine path with zero new code required.

**English-only variants:**

| Model | Params | INT8 Size | Notes |
|-------|--------|-----------|-------|
| distil-small.en | ~166M | ~298 MB | Encoder ~103 MB + decoder ~195 MB |
| distil-medium.en | ~394M | ~197 MB | Middle ground |
| distil-large-v3 | 756M | ~378 MB | English + multilingual (degraded non-EN) |
| distil-large-v3.5 | 756M | ~378 MB | Latest, 51% smaller than large-v3, 1.5x faster than turbo |

**Accuracy vs whisper-large-v3 (distil-large-v3):**
- Short-form: 9.7% vs 8.4% WER
- 6.3x faster inference, 1.5x lower hallucination rate

**sherpa-onnx integration:** Identical to `OfflineWhisperModelConfig` — no new engine code. Pre-converted models available under csukuangfj.

**License:** Apache 2.0

**Assessment:** Best option for users who want fast English dictation and already have a Whisper model downloaded — the familiar model list entry but faster. distil-small.en at ~298MB sits between whisper-small (~170MB) and Turbo (~1GB), offering large-v2-distilled accuracy at roughly 3x smaller than Turbo. Multilingual distil-large-v3 is not worth it over standard large-v3 for non-English use.

---

### NeMo FastConformer CTC (NVIDIA)

Pure CTC decoder on the FastConformer encoder. Faster per-inference than the encoder-decoder Canary (no autoregressive decode). Multiple English sizes plus a 10-language European variant.

**English models:**

| Model | INT8 Size | WER (LS clean) | WER (LS other) |
|-------|-----------|----------------|----------------|
| Conformer-CTC Small | ~44 MB | ~3–4% | — |
| Conformer-CTC Medium | ~64 MB | ~2.8% | — |
| Conformer-CTC Large | ~162 MB | 2.2% | 4.3% |
| Parakeet TDT-CTC 110M | ~55 MB | ~2–3% | — |

**Multilingual:** `sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k` — Belarusian, German, English, Spanish, French, Croatian, Italian, Polish, Russian, Ukrainian. No INT8 conversion confirmed yet.

**sherpa-onnx integration:** `OfflineNemoEncDecCtcModelConfig` (field `nemo_ctc` in `OfflineModelConfig`). Pre-converted: `csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-{small,medium,large}`.

**License:** Apache 2.0 (NVIDIA NeMo)

**Assessment:** Conformer-CTC Small/Medium are compelling — 44–64MB with Whisper-base-class WER, CTC so no autoregressive penalty. Large at 162MB reaches 2.2% clean, competitive with Parakeet at less than a quarter of the size. The 10-language model needs INT8 conversions before it's mobile-viable.

---

### Zipformer (k2/icefall)

The native architecture of the sherpa-onnx project. Ships as either transducer (reuses the Parakeet engine path) or CTC. Streaming and offline variants. Extensive language coverage with regularly updated pre-converted INT8 models.

**Languages with pre-converted INT8 models:**
- English (multiple sizes, LibriSpeech + GigaSpeech trained)
- Mandarin Chinese (standalone, bilingual zh-en, trilingual zh-en-Cantonese)
- Korean, French, Bengali, Vietnamese, Russian
- Cantonese, Wu dialect (newer 2025–2026 additions)

**Size range:** Tiny streaming English models start at ~20MB. Vietnamese INT8 is ~74MB. Larger models under 300MB.

**WER:** Competitive with Whisper base/small for English. Chinese models trained on WenetSpeech/AISHELL are substantially better than Whisper for Mandarin.

**sherpa-onnx integration:**
- Transducer variant: same `OfflineTransducerModelConfig` fields as Parakeet, but requires `model_type = "transducer"` (not `"nemo_transducer"`). Currently `transcribe.rs` hardcodes `"nemo_transducer"` for all `ModelEngine::Transducer` — would need a `model_type` field added to that variant, or a separate enum variant.
- CTC variant: `OfflineZipformerCtcModelConfig` (field `zipformer_ctc`, already in sherpa-onnx)

**License:** Apache 2.0 (k2/icefall)

**Assessment:** High value, especially the tiny streaming English models (20–50MB). The transducer variants need a one-line change to `ModelEngine::Transducer` to carry the model type string through. Good coverage for Korean, Vietnamese, Bengali where Whisper is the only current option and is bloated. The streaming variants are the most interesting for Android latency.

---

### FireRedASR2 (FireRedTeam / Kuaishou)

Conformer-based Chinese ASR with dialect coverage. Two variants: AED (encoder-decoder, ~1B params) and CTC (faster, mobile-viable). Android APK confirmed working.

| Variant | INT8 Size | Notes |
|---------|-----------|-------|
| FireRedASR2-CTC | ~300–400 MB est. | Mobile-viable |
| FireRedASR2-AED | ~500–700 MB est. | Higher accuracy, larger |

**Languages:** Mandarin + English + 20+ Chinese dialects — Cantonese (Yue), Sichuan, Wu (Shanghai), Fujian (Min), and more.

**Accuracy (Mandarin benchmarks, AED model):**
- Avg-Mandarin-4: 2.89% CER — outperforms Doubao-ASR, Qwen3-ASR, FunASR

**sherpa-onnx integration:** `OfflineFireRedAsrModelConfig` (encoder + decoder fields). Pre-converted: `csukuangfj/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25`.

**License:** Apache 2.0

**Assessment:** Best option for Mandarin + Chinese dialect coverage. SenseVoice supports Cantonese but FireRedASR2 covers 20+ dialects including regional varieties SenseVoice does not. CTC variant is Android-confirmed and faster than AED. Primary use case: Chinese regional language support.

---

### Meta Omnilingual ASR (Meta AI, Nov 2025)

Self-supervised encoder (wav2vec lineage) with CTC head. The standout feature is the number of supported languages: 1,600+, including 500+ with no prior ASR support. Language-conditioned via a token passed at inference.

| Model | Params | INT8 Size | Notes |
|-------|--------|-----------|-------|
| 300M CTC | 300M | ~348 MB | Android APK confirmed |
| 1B CTC | 1B | ~500 MB est. | Higher accuracy |

**Accuracy:** CER below 10% for 78% of supported languages at the 7B tier. The 300M is degraded but still functional across all 1,600 languages. For major languages (English, Spanish, French, Arabic, etc.) accuracy is competitive with similarly-sized models.

**sherpa-onnx integration:** `OfflineOmnilingualAsrModelConfig` (encoder + decoder fields, similar structure to FireRedASR). Pre-converted: `csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12` (348 MB).

**License:** Apache 2.0 (model) + CC BY 4.0 (training data)

**Assessment:** Unique value proposition — no other model covers obscure languages. For major European/Asian languages the accuracy trails dedicated models, so this is additive rather than a replacement. The 300M at 348MB is within the mobile constraint. The UI would need a language selector since the language token is required at inference time.

---

### Moonshine (Useful Sensors)

Encoder-decoder transformer designed for on-device use. Variable-length input (no 30-second padding like Whisper). Uses RoPE instead of absolute position embeddings.

**V1 (Oct 2024) — English only, MIT license:**

| Model | Params | INT8 Size | WER (LS clean) | WER (LS other) |
|-------|--------|-----------|---------------|---------------|
| Tiny | 27M | ~118 MB | 4.52% | 11.71% |
| Base | 62M | ~273 MB | 3.23% | 8.18% |

V1 Tiny beats Whisper tiny.en despite fewer parameters. V1 Base beats Whisper base.en across the board.

**V2 "Flavors of Moonshine" (Sep 2025) — per-language models:**
Same tiny/base sizes, individually trained for Arabic, Chinese, Japanese, Korean, Ukrainian, Vietnamese, Spanish. Claims 48% lower error than Whisper Tiny on average. License: permissive ("other").

**Streaming (Jan 2026) — English only, MIT license:**

| Model | Params | RPi5 Latency (10s audio) | WER (LS clean) | WER (LS other) |
|-------|--------|--------------------------|---------------|---------------|
| Tiny | 34M | 237ms | 4.49% | 12.09% |
| Small | 123M | 527ms | 2.49% | 6.78% |
| Medium | 245M | 802ms | 2.08% | 5.00% |

Streaming Medium beats Whisper Large v3 on both accuracy and speed on RPi5. No sherpa-onnx ONNX conversions exist yet for streaming models.

**sherpa-onnx integration:** `OfflineMoonshineModelConfig`. 4–5 ONNX files per model. Pre-converted INT8: `csukuangfj/sherpa-onnx-moonshine-tiny-en-int8`, `csukuangfj/sherpa-onnx-moonshine-base-en-int8`, V2 per-language repos under `csukuangfj2/`.

**Assessment:** Strong candidate for a lightweight desktop option. Tiny at 118 MB offers Whisper-base accuracy at a fraction of the size and dramatically faster on ARM. On mobile, still encoder-decoder (autoregressive decode) so Parakeet's transducer will be faster per inference, but Moonshine's smaller size means faster loading and lower memory. Worth testing on Android.

---

### Canary (NVIDIA NeMo)

Encoder-decoder using FastConformer encoder (same as Parakeet) with autoregressive Transformer decoder. Same NeMo team as Parakeet. CC-BY-4.0 license. Supports translation natively.

| Model | Params | Languages | INT8 Size |
|-------|--------|-----------|-----------|
| 180M Flash | 182M | en/de/es/fr + translation | ~207 MB |
| 1B Flash | 883M | en/de/es/fr + translation | — |
| 1B v2 | 978M | 25 European + translation | — |
| Qwen 2.5B | 2.5B | English only (LLM post-proc) | — |

**English accuracy (Open ASR Leaderboard mean WER):**

| Model | Mean WER | RTFx |
|-------|---------|------|
| Parakeet TDT 0.6B v2 | 6.05% | 3380x |
| Canary 1B Flash | 6.35% | 1046x |
| Canary 180M Flash | ~7.07% | 1233x |
| Canary Qwen 2.5B | 5.63% | 418x |

**sherpa-onnx integration:** `offline_canary`. Only 180M Flash has ONNX conversions: `csukuangfj/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8` (207 MB).

**Assessment:** Canary 180M Flash at 207 MB is the interesting one — small, 4-language multilingual, translation support. English accuracy trails Parakeet but beats Whisper at similar size. Autoregressive decoder means slower mobile inference. Best suited for desktop multilingual use.

---

### SenseVoice (Alibaba / FunAudioLLM)

Non-autoregressive encoder-only model. Single forward pass, no sequential decoding. Claims 5x faster than Whisper Small, 15x faster than Whisper Large.

| Model | Params | Languages | INT8 Size |
|-------|--------|-----------|-----------|
| Small | 234M | 50+ (best: zh/en/ja/ko/yue) | ~228 MB |

Extra features: speech emotion recognition, audio event detection (BGM, applause, laughter), spoken language ID, built-in ITN.

English accuracy roughly matches Whisper Small. Substantially better on Mandarin/CJK (e.g. 2.96% vs 10.04% WER on AISHELL-1).

**sherpa-onnx integration:** `OfflineSenseVoiceModelConfig`. Single ONNX file + tokens. Pre-converted: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` (228 MB).

**License:** FunASR Model Open Source License v1.1. Commercial use permitted with attribution. Contains a clause revoking the license if you publicly denigrate the project — unusual, worth noting.

**Assessment:** Best option for CJK language support. Non-autoregressive means it should be fast on mobile. English accuracy matches Whisper Small but trails Parakeet. Primary value for Chinese/Japanese/Korean/Cantonese users. For Mandarin + Chinese dialect depth, FireRedASR2 is a stronger alternative.

---

### GigaAM v3 (Sberbank / Salute Developers)

Conformer-based model for Russian. Pre-trained on 700K hours of Russian speech. MIT license. Claims 50% better than Whisper large-v3 on Russian.

| Variant | Type | Mean WER (Russian) |
|---------|------|-------------------|
| CTC | CTC | 9.2% |
| RNNT | RNN-Transducer | 8.4% |
| E2E RNNT | RNNT + punctuation/ITN | 8.4% |

**sherpa-onnx integration:** CTC via `nemo_ctc`, RNNT via existing `transducer` engine (same as Parakeet — straightforward integration). Pre-converted float32 only, no INT8 yet: `sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16`, `sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16`.

**Assessment:** Only relevant for Russian. MIT license is clean. RNNT variant uses the existing transducer engine so integration would be straightforward once INT8 quantized versions appear.

---

### Breeze ASR

Optimised for Taiwanese Mandarin with code-switching support. ~1.1 GB download. Niche use case. No further research done.

---

### Priority summary

**High priority — zero or minimal new engine code:**
- Distil-Whisper small.en INT8 (~298 MB) — English, existing Whisper path, zero code change; large-v2-distilled accuracy, 3x smaller than Turbo
- NeMo Conformer-CTC Small INT8 (~44 MB) — tiny English CTC, needs NeMo CTC engine support
- NeMo Conformer-CTC Large INT8 (~162 MB) — 2.2% WER, best accuracy-per-MB in English
- Zipformer streaming English (~20–50 MB) — transducer path, needs `model_type` string threaded through `ModelEngine::Transducer`; streaming support
- Parakeet TDT 0.6B v2/v3 — already in the model registry (v2: best English accuracy 1.69% clean; v3: 25 European languages)

**High priority — significant language coverage, new engine types:**
- Meta Omnilingual ASR 300M INT8 (~348 MB) — 1,600+ languages, unique coverage
- FireRedASR2-CTC INT8 (~300–400 MB) — Mandarin + 20 Chinese dialects, Android confirmed

**Medium priority:**
- Moonshine Tiny INT8 (~118 MB) — lightweight English, very fast
- Moonshine Base INT8 (~273 MB) — better accuracy, still compact
- Canary 180M Flash INT8 (~207 MB) — 4-language multilingual + translation
- SenseVoice INT8 (~228 MB) — CJK, non-autoregressive speed
- GigaAM v3 (~152 MB) — best Russian support

**Wait for conversions:**
- Moonshine Streaming — would be excellent for mobile but no ONNX conversions yet
- Canary 1B v2 — 25 European languages + translation but no sherpa-onnx conversion
- NeMo FastConformer CTC 10-language — no INT8 conversion confirmed yet
- GigaAM v3 INT8 — float32 only currently

**Engine types required (current codebase supports `whisper` and `parakeet`/transducer):**
- `nemo_ctc`: NeMo Conformer-CTC models, GigaAM v3 CTC, Parakeet CTC 110M
- `zipformer_ctc`: Zipformer CTC (transducer variant is free via existing Parakeet path)
- `moonshine`: Moonshine V1/V2
- `canary`: Canary models
- `sensevoice`: SenseVoice
- `fire_red_asr`: FireRedASR2
- `omnilingual`: Meta Omnilingual ASR

---

## Evaluated — not viable for this stack

### Voxtral (Mistral)

Released mid-2025, Apache 2.0. Two ASR variants: Mini (3B, on Ministral 3B) and Small (24B). Claims to outperform Whisper large-v3. ONNX conversion exists (`onnx-community/Voxtral-Mini-3B-2507-ONNX`) with 4-bit and 8-bit quantisation options.

**Android: not viable.** Mini 3B memory: encoder ~327 MB, decoder ~3,047 MB, total ~3.4 GB. The decoder alone exceeds what most Android devices can give a single app.

**Desktop: viable with caveats.** Memory fine on 8–16 GB machines. Speed unknown — LLM decoder architectures generate token-by-token, slower than CTC/transducer on CPU. Also, sherpa-onnx does not support Voxtral on any platform. A second inference backend (llama.cpp or candle) would be required, which is a meaningful engineering investment.

**Watch:** sherpa-onnx adds Voxtral support; Mistral ships a distilled/quantized variant under ~500 MB total; community llama.cpp port with validated real-time CPU benchmarks.

---

### faster-whisper (SYSTRAN)

Reimplementation of Whisper using CTranslate2. Claims 4x faster than the original `openai/whisper` Python package with INT8. The comparison baseline is slow — the app already runs Whisper via sherpa-onnx ONNX models which achieves similar improvements without added dependency.

**Android: not viable.** Python-first. CTranslate2 supports ARM64 at the C++ level but no Android NDK build path or JNI bindings exist.

**Desktop:** useful for Python server/desktop apps. For a Rust/Tauri app, adding CTranslate2 alongside sherpa-onnx means two separate native inference stacks with incompatible model formats. Solves a problem we don't have.

---

### MMS (Meta Massively Multilingual Speech)

Wav2vec 2.0 backbone with per-language adapter layers. Covers 1,100+ languages.

**Not viable:** No sherpa-onnx integration and no official pre-converted models. The per-language adapter architecture makes it awkward to bundle — each language requires different weights. Meta Omnilingual ASR covers more languages (1,600+) with full sherpa-onnx support, making MMS redundant.

---

### Android native STT

Two platform-provided options:

**SpeechRecognizer API (Android 12+):** On-device mode quality varies by manufacturer, generally below Whisper Small. Cloud-backed mode sends audio to Google. Callback-based Java API — JNI plumbing required from Rust. No word timestamps, no confidence scores.

**ML Kit GenAI / Gemini Nano (announced May 2025):** Significantly better quality. ~1 GB download. As of April 2026: Pixel 10 only. Too narrow for a general app.

**Potential use case for SpeechRecognizer:** zero-download fallback on first launch before the user has downloaded any model. Already on every Android 12+ device, routes through existing delivery pipeline via a Kotlin/JNI wrapper. Accuracy and privacy caveats would need to be surfaced in the UI.

**Watch:** ML Kit GenAI expands to mainstream Android (Pixel 9+, Samsung flagships). When that happens the zero-download story becomes compelling — no model management, Google maintains quality.

---

## Parakeet V2 vs V3 — English accuracy comparison

Researched April 2026. Sources: NVIDIA model cards, HuggingFace Open ASR Leaderboard, arXiv:2509.14128.

### WER on HuggingFace Open ASR Leaderboard (8 English benchmarks)

| Benchmark | V2 | V3 | Delta |
|-----------|----|----|-------|
| LibriSpeech test-clean | 1.69% | 1.93% | +0.24pp |
| LibriSpeech test-other | 3.19% | 3.59% | +0.40pp |
| AMI | 11.16% | 11.31% | +0.15pp |
| Earnings-22 | 11.15% | 11.42% | +0.27pp |
| GigaSpeech | 9.74% | 9.59% | -0.15pp |
| SPGI Speech | 2.17% | 3.97% | +1.80pp |
| TEDLIUM-v3 | 3.38% | 2.75% | -0.63pp |
| VoxPopuli | 5.95% | 6.14% | +0.19pp |
| **Average** | **6.05%** | **6.34%** | **+0.29pp** |

V2 wins on 6 of 8 benchmarks. The SPGI Speech regression in V3 is the most notable — 83% relative degradation. SPGI Speech is business/financial speech, which is close to the technical dictation use case this app targets.

V3 wins on TEDLIUM (lecture/talk speech) and GigaSpeech (podcasts/YouTube), suggesting its broader training data helps with informal spontaneous speech but hurts on read and professional speech.

### Why V3 regresses on English

V2 was pretrained on LibriLight using a wav2vec-style SSL objective — heavily English-focused. V3 was initialized from a multilingual CTC checkpoint (Granary dataset, 25 languages), which dilutes English specificity. NVIDIA traded English accuracy for multilingual coverage.

### Verdict for English-only use

V2 is the better default. V3 only adds value if multilingual support is needed. The registry description calling V3 "latest" is misleading — it is newer but not better for English.

**INT8 sizes are comparable:** V2-int8 ~630 MB, V3-int8 ~670 MB. No storage reason to prefer V3.

---

## Best English ASR models for on-device inference (April 2026)

### Leaderboard context

| Model | Params | LS clean | LS other | Avg WER | On-device? |
|-------|--------|----------|----------|---------|------------|
| Canary-Qwen-2.5B | 2.5B | 1.60% | 3.10% | 5.63% | No (GPU only) |
| IBM Granite-Speech-3.3-8B | 8B | — | — | 5.74% | No (GPU only) |
| **Parakeet TDT 0.6B V2** | **0.6B** | **1.69%** | **3.19%** | **6.05%** | **Yes** |
| Parakeet TDT 0.6B V3 | 0.6B | 1.93% | 3.59% | 6.34% | Yes |
| Moonshine Streaming Medium | 245M | 2.08% | 5.00% | ~6.65% | Yes |
| Distil-Whisper large-v3.5 | 756M | — | — | 7.21% | Yes |
| Whisper large-v3 | 1.55B | ~2.7% | ~5.2% | 7.44% | Yes (slow on CPU) |
| Whisper large-v3-turbo | 809M | ~2.0% | ~4.0% | ~8.0–8.5% | Yes |

### Conclusion

**Parakeet TDT 0.6B V2 is the best on-device English ASR model available.** No sub-1B model beats it. The only models with lower WER (Canary-Qwen-2.5B, Granite-Speech-3.3-8B) require NVIDIA GPU and are not viable on Android or CPU-only desktop.

Moonshine Streaming Medium (245M) comes close at ~6.65% average and is purpose-built for edge devices, but no sherpa-onnx ONNX conversions exist yet for the streaming variants. Worth revisiting when they appear.

**Against Whisper large-v3-turbo specifically:** Parakeet V2 is ~2pp more accurate on average, 25% smaller, and substantially faster (non-autoregressive transducer vs autoregressive decoder). The only reasons to prefer turbo are multilingual support (99 languages) and broader ecosystem compatibility.
