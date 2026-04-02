# Fluency Correction Research

The goal is to emulate Wispr Flow's post-processing step: take the raw transcription and produce text that reads like something a person would write, not just a literal capture of what was said. Flow does this with a cloud LLM. We want to do it locally on Android.

This is distinct from the grammar correction work (nlprule, GECToR). Grammar correction catches known error patterns — a/an agreement, homophones, subject-verb agreement. Fluency correction asks a different question: does this sentence make sense? Could it have been transcribed wrong in a way that rule-matching wouldn't catch?

---

## The Latency Problem

Autoregressive LLMs cannot hit a sub-200ms target on mobile CPU. Every output token requires a full forward pass:

| Model | Quant | Est. ARM64 latency (paragraph) |
|-------|-------|-------------------------------|
| SmolLM2 135M | FP32 | ~870ms |
| Qwen 2.5 0.5B | Q4 | ~1.6s |
| Qwen 2.5 3B | Q4 | ~4-5s |

These benchmarks are PyTorch FP32. ONNX INT8 gives roughly 3-5x speedup, putting SmolLM2 135M at ~175-300ms — borderline but still not reliably under 200ms for longer sentences.

The only models that reliably hit 200ms are non-generative: single forward pass classifiers and taggers.

---

## Models Evaluated and Eliminated

### EdgeFormer (Microsoft Editor)

11M params, ~11MB INT8, F0.5=52.7 on CoNLL-14, MIT license. Was the most promising option.

**Status: inaccessible.** The pretrained checkpoint URL on Microsoft blob storage now returns Access Denied (confirmed April 2026, GitHub issue unilm#1323). The ONNX inference release promised in September 2022 (unilm#860) was never delivered. The model exists in the paper but cannot be obtained.

### CoEdIT (Grammarly, 2023)

Flan-T5 fine-tuned for instruction-guided text editing. ONNX weights available via `rayliuca` on HuggingFace.

**Eliminated on two counts:**
- License: CC-BY-NC 4.0 on all variants — commercial use not permitted.
- Latency: even the smallest available variant (flan-t5-base, ~422MB INT8) is 300–700ms on ARM64. coedit-large INT8 (~1.24GB) is 2–5 seconds. Both exceed the budget by an order of magnitude.

No `coedit-small` variant exists. The base is the smallest released.

### LaserTagger (Google, 2019)

BERT-based edit-operation tagger that predicts KEEP/DELETE/INSERT per token in a single forward pass. The constrained output space means no autoregressive decoding. The concept is sound — one BERT-base forward pass is ~40-60ms INT8 on ARM64.

**Eliminated: no English correction checkpoint exists.** The repo provides training code and data scripts only. The single HuggingFace checkpoint (`maxliaops/lasertagger-chinese`) is Chinese only. Original codebase is TensorFlow 1.x with no official PyTorch port. Training from scratch on BEA-2019/JFLEG would require GPU budget and several weeks of work. Worth revisiting if a suitable pretrained English checkpoint appears.

### FELIX (Google, 2020)

Google's follow-up to LaserTagger with a pointer mechanism for more flexible insertions. Same TF-based codebase. No released English inference weights found. GitHub repository appears to have been removed or relocated. Same conclusion as LaserTagger.

---

## The Routing Architecture

The key insight from this research: a two-stage approach lets the fast path stay fast while enabling heavier correction only when needed.

Originally designed with DistilGPT2 as the router (per-token log-probs to localize spans). Empirical testing showed DistilGPT2 is unworkable for technical text — see Component 1 section. The architecture below uses a CoLA classifier instead, which trades span localization for reliable routing.

```
Transcription output
        │
        ▼
CoLA classifier  (~25ms, single forward pass)
        │
        ├── P(acceptable) ≥ threshold ──► pass through unchanged
        │
        └── P(acceptable) < threshold ──► full sentence correction
                                                │
                                                ▼
                                   t5-efficient-tiny correction
                                   (~50ms ONNX INT8)
                                                │
                                                ▼
                                          Final output
```

The full-sentence correction (vs span-only) is a trade-off: simpler to implement, but t5-efficient-tiny runs on the whole input even when only one word is wrong. The router ensures this only happens for sentences that need it.

---

## Component 1: DistilGPT2 Perplexity Scoring

**Why DistilGPT2 over a CoLA sentence classifier:**

A CoLA classifier (e.g. ELECTRA-small fine-tuned on CoLA, ~14MB INT8, 5-10ms) gives a binary sentence-level judgment: fluent or not fluent. It tells you something is wrong but not where.

DistilGPT2 gives a **log-probability per token** in one forward pass. Low log-prob tokens are the likely wrong words — the specific spans that need correction. This is the information you need to do targeted fixing rather than a full sentence rewrite.

| Property | Value |
|----------|-------|
| Parameters | 88M |
| FP32 size | ~352MB |
| INT8 size | ~88MB |
| Inference mode | Single forward pass (no generation) |
| Est. ARM64 latency | 20-40ms |
| License | Apache 2.0 |
| ONNX | Available via Xenova/transformers.js ecosystem |

**How it works in the pipeline:**

Run the full transcription through DistilGPT2 as a language model scoring pass — no tokens are generated, just log-probabilities extracted. Tokens below a threshold (empirically tuned, probably around the 10th percentile of the training distribution) are flagged. Contiguous or near-contiguous flagged tokens form a span for correction.

If no tokens are flagged, the sentence passes through with zero additional latency beyond the 20-40ms scoring pass.

**Threshold tuning:** The threshold should be calibrated on real dictation output, not clean text. ASR output has specific failure modes (homophones, dropped articles, wrong verb form) that differ from general text noise. The threshold needs to be set high enough to catch real errors without flagging correct low-frequency words.

**Empirical result — fatal flaw for technical text:** Tested at thresholds -7.0 through -9.5 using `scripts/test_pipeline.py`. Technical vocabulary gets the same low log-probs as genuine errors regardless of threshold:

- `deployed` → -16.24
- `PR` → -12.23
- `function` → -11.83
- `REST-compliant` → flagged at every tested threshold

At -7.0 (default): TP=15 TN=0 FP=8 FN=0. Routes 100% of sentences including all clean technical text.
At -9.5 (permissive): TP=10 TN=0 FP=8 FN=5. Still routes all 8 clean technical sentences.

TN=0 at every usable threshold. DistilGPT2 is unworkable as a router for tech-heavy dictation because it confuses rare vocabulary with grammatical errors. The per-token localization advantage is real, but irrelevant if the router cannot distinguish correct from incorrect.

---

## Component 2: t5-efficient-tiny Grammar Correction

**Model:** `visheratin/t5-efficient-tiny-grammar-correction`

| Property | Value |
|----------|-------|
| Base model | google/t5-efficient-tiny (15.58M params) |
| FP32 size | ~62MB |
| INT8 encoder | ~11.5MB |
| Full INT8 ONNX (enc + dec + dec-init) | ~44MB |
| Est. ARM64 latency (full sentence) | 50-150ms |
| License | MIT |
| ONNX | Available — quantized ONNX already built by the author |

ONNX weights are already published and quantized. The encoder is only 11.5MB INT8, which is unusually small for a transformer encoder.

**Important caveat:** The model was trained on the C4_200M dataset, which is a fluency rewrite corpus derived from Common Crawl — not a formal GEC dataset like BEA-2019 or CoNLL-14. This means:

- It will smooth awkward phrasing and improve flow
- It may rewrite correct but informal phrasing into something different from what the user intended
- There are no published F0.5 scores against standard GEC benchmarks, so quality is hard to quantify
- It needs benchmarking against real dictation samples before committing to it

The `t5-efficient-mini` variant (31M params, ~77MB INT8) is available at the same URL and offers higher quality at roughly 2x the latency — worth benchmarking alongside tiny.

---

## Alternative Stage 1: CoLA Sentence Classifier

If the goal is only to decide whether to skip the correction pass entirely (not to localize spans), a CoLA-trained classifier is faster and smaller than DistilGPT2.

CoLA (Corpus of Linguistic Acceptability) models are trained on grammatical acceptability judgments rather than word frequency. This means they learn structural grammar, not how common a word is. Technical vocabulary (`API`, `CI`, `REST-compliant`, `deployed`) scores as acceptable because it is grammatically valid in context.

| Model | Params | INT8 size | Est. ARM64 latency | License | ONNX ready |
|-------|--------|-----------|-------------------|---------|------------|
| pszemraj/electra-small-discriminator-CoLA | 13.5M | ~14MB | 5-10ms | Apache 2.0 | No |
| pszemraj/xtremedistil-l12-h384-uncased-CoLA | 33M | ~35MB | 10-20ms | MIT | model.onnx exists |
| pszemraj/deberta-v3-xsmall-CoLA | 70.8M | ~70MB | 20-40ms | MIT | model.onnx exists |
| pszemraj/electra-base-discriminator-CoLA | 110M | ~110MB | 15-25ms | Apache 2.0 | Yes |

This is a simpler pipeline (classify → correct whole sentence if flagged vs. pass through) at the cost of losing span-level localization.

**Empirical results — tested via `scripts/test_cola_router.py` and `scripts/test_routers_v2.py`:**

Test set: 15 error cases, 8 clean cases.

| Model | Threshold | TP | TN | FP | FN | F1 | Scorer ms |
|-------|-----------|----|----|----|----|-----|-----------|
| xtremedistil | 0.85 | 6 | 6 | 2 | 9 | 0.52 | 7ms |
| electra-base | 0.85 | 8 | 7 | 1 | 7 | 0.67 | 17ms |
| electra-small | 0.85 | 10 | 7 | 1 | 5 | 0.77 | 5ms |
| deberta-v3-xsmall | 0.85 | 10 | 8 | 0 | 5 | 0.80 | 26ms |
| **electra-small** | **0.80** | **10** | **8** | **0** | **5** | **0.80** | **4ms** |

**Key finding:** electra-small at threshold=0.80 is identical to deberta-v3-xsmall at 0.85 in routing accuracy, but runs in **4ms vs 26ms** and is 13.5M vs 70.8M params. The only FP at threshold=0.85 was `API endpoints need to be REST-compliant` (p=0.803), which falls outside the 0.80 threshold.

Counterintuitively, electra-base (110M) performs *worse* than electra-small (13.5M), scoring F1=0.67. It's extremely conservative — misses `their is a problem`, `could of been a lot worse`, `last week we push the release`. Bigger is not better here.

**Threshold sweep for electra-small** (`scripts/test_router_threshold_sweep.py`):

There is a natural gap between the highest-p routable fix case (`could of been a lot worse`, p=0.684) and the lowest-p clean sentence (`API endpoints need to be REST-compliant`, p=0.803). Any threshold in the range [0.70, 0.80] gives identical routing: TP=10 TN=8 FP=0 FN=5.

| Threshold | TP | TN | FP | FN | F1 |
|-----------|----|----|----|----|-----|
| 0.60 | 7 | 8 | 0 | 8 | 0.64 |
| 0.65 | 9 | 8 | 0 | 6 | 0.75 |
| 0.70–0.80 | **10** | **8** | **0** | **5** | **0.80** |
| 0.85 | 10 | 7 | 1 | 5 | 0.77 |
| 0.90 | 12 | 5 | 3 | 3 | 0.80 |
| 0.95 | 15 | 0 | 8 | 0 | 0.79 |

Production recommendation: **threshold=0.75** — sits in the middle of the safe gap, and provides more margin against distribution shift than 0.80. Threshold=0.90 catches 2 more errors but routes 3 clean sentences unnecessarily; the tradeoff is unfavorable.

The 5 remaining false negatives across all models are the same structurally-valid errors:
- `your going to love this new feature`
- `its going to take longer than expected`
- `I should of called them earlier`
- `could of been a lot worse`
- `earlier today I fix the bug and deploy it`

These parse as syntactically valid — CoLA judges grammatical structure, not lexical choice. `your going to` is structurally `PRON VERB PARTICLE VERB` which is acceptable to a structure-level model. Catching these requires semantic/lexical context, not acceptability scoring.

---

## Corrector Quality Comparison

Tested via `scripts/test_fluency_correction.py`, `scripts/test_correctors_v2.py`, and `scripts/test_correctors_v3.py` on 23 ASR error cases (15 fix + 8 preserve). PyTorch CPU times; ONNX INT8 estimated roughly half.

| Model | Params | License | Trained on | Fixed | OverCorr | Median ms |
|-------|--------|---------|-----------|-------|----------|-----------|
| visheratin/t5-efficient-tiny | 15.6M | MIT | C4 fluency rewrites | 15/15 | 4/8 | 50ms |
| visheratin/t5-efficient-mini | 31.2M | MIT | C4 fluency rewrites | 15/15 | 6/8 | 79ms |
| Unbabel/gec-t5_small | 60M | Apache 2.0 | CoNLL-13/14 GEC | 15/15 | 3/8 | 103ms |
| pszemraj/grammar-synthesis-small | 77M | Apache 2.0 | JFLEG | 14/15 | 6/8 | 216ms |
| gotutiyan/gec-bart-base | 100M | MIT | C4+BEA19 | 15/15 | 7/8 | 175ms |
| Buntan/gec-t5-v1_1-small | 77M | Apache 2.0 | cLang-8 | — | — | — |

All models fix every error except grammar-synthesis-small (misses 1). The differentiator is over-correction of clean text.

### Eliminated candidates

**pszemraj/grammar-synthesis-small** — JFLEG training teaches fluency *rewrites*, not minimal error correction. The model makes semantic changes to clean text:
- `"The quick brown fox"` → `"The yellow fox"` (changed "brown" to "yellow")
- `"API endpoints need to be REST-compliant"` → `"API ends need to be REST-compliant"` (truncated "endpoints")
- `"The PR is in review and CI is green"` → `"The PR is in review and is green"` (dropped "CI is")
- `"The function returns a list of strings"` → `"a list of string values"` (inserted "values")
- Also misses `"your going to"` (the only model to not fix it)

Do not use JFLEG-trained models for this task. JFLEG rewrites fluency; it does not target minimal, safe error correction.

**gotutiyan/gec-bart-base** — 7/8 over-corrections, worst result. BART generation adds trailing periods to everything. Also changes meaning: `"new API endpoint yesterday"` → `"new API backend yesterday"` (semantic corruption). MIT license but quality disqualifies it.

**Buntan/gec-t5-v1_1-small** — Echoes the `"gec: "` prefix verbatim in output (e.g., output starts with `"gec: ..."` or `"Gec: ..."`). Model was not trained to strip its own prefix. 8/8 preserve cases "changed" because the prefix is appended to every output. Unusable.

### t5-efficient-tiny (15.6M, MIT) — fast baseline
4/8 over-corrections:
- `"Please send me the report by end of day"` → added "the end of the day" (minor)
- `"we remove old logs every Friday"` → capitalized only (trivial)
- `"The PR is in review and CI is green"` → `"The PR is in the review and CI is green"` (inserted "the")
- `"API endpoints need to be REST-compliant"` → `"The API endpoints need to be REST-compliant"` (inserted "The")

Notable fix quality issues:
- `"your going to love this new feature"` → `"Are you going to love this new feature?"` (converted to question)
- `"I should of called them earlier"` → `"I should call them earlier"` (dropped "have", changed tense)

Has published ONNX weights (~44MB INT8), already quantized by the author. Fastest ONNX-ready option.

### t5-efficient-mini (31.2M, MIT) — not recommended
6/8 over-corrections — worse than tiny on clean text despite being 2x larger.
- `"We deployed the new API endpoint yesterday"` → `"We deployed a new API endpoint yesterday"` (changed `the` to `a`)
- `"The function returns a list of strings"` → `"a list of the strings"`
- `"he walk to the office every day"` → `"He walked to the office every day"` (wrong: present→past)

More aggressive rewriting, not more accurate. Actively a step down.

### Unbabel/gec-t5_small (60M, Apache 2.0) — best corrector found
3/8 over-corrections, most conservative. Trained on actual annotated GEC data (CoNLL-13/14) rather than synthetic noise or fluency rewrites.

3 over-corrections (all minor):
- `"Please send me the report by end of day"` → added "the end of the day"
- `"we remove old logs every Friday"` → capitalized first letter only
- `"The PR is in review and CI is green"` → added "the" before CI

All 4 clean technical sentences unchanged:
- `"We deployed the new API endpoint yesterday"` — unchanged
- `"The function returns a list of strings"` — unchanged
- `"API endpoints need to be REST-compliant"` — unchanged
- `"The quick brown fox..."` — unchanged

Fix quality is better than tiny in all categories tested:
- `"your going to"` → `"You're going to"` (correct, vs tiny's question conversion)
- `"I should of called"` → `"I should have called"` (correct, vs tiny's "I should call")
- All homophones, tense, agreement, and ASR duplicates corrected correctly

**Downside:** No published ONNX weights. Requires `"gec: "` prefix on inputs.

**ONNX export confirmed working** via `scripts/export_gec_t5_onnx.py`. Exports three files:

| File | FP32 | INT8 |
|------|------|------|
| encoder_model.onnx | 135MB | 34MB |
| decoder_model.onnx | 222MB | 56MB |
| decoder_with_past_model.onnx | 210MB | 53MB |
| **Total** | **567MB** | **~143MB** |

Output quality verified 8/8 match between PyTorch, FP32 ONNX, and INT8 ONNX on the test set. The Python 3.14 + Optimum 2.1.0 bug (`NormalizedConfig.__init__() got multiple values for argument 'allow_new'`) is patched at the top of the export script — Python 3.14 treats `functools.partial`-wrapped classes as descriptors, prepending the instance as a positional arg. Fix: access `NORMALIZED_CONFIG_CLASS` via `type(self)` rather than `self` to bypass descriptor binding.

x86 ONNX latency was 1245ms median (vs 135ms PyTorch) — an artifact of unoptimized ORT on x86. ARM64 ORT with NEON kernels is expected to be ~65ms INT8.

## Full Pipeline Test Results

Tested via `scripts/test_pipeline_v2.py`: 220 sentences (130 fix, 90 preserve), electra-small router at threshold=0.75 + Unbabel/gec-t5_small corrector.

### Router results

| Metric | Value |
|--------|-------|
| TP | 95 / 130 |
| TN | 78 / 90 |
| FP | 12 / 90 |
| FN | 35 / 130 |
| F1 | 0.80 |
| Routed to corrector | 107 / 220 |
| Router latency | 9ms median |

Per-category routing breakdown:

| Category | Total | Routed | Missed (FN) or False-routed (FP) |
|----------|-------|--------|----------------------------------|
| fix/asr-dup | 15 | 15 | FN=0 |
| fix/agreement | 20 | 18 | FN=2 |
| fix/plural | 10 | 8 | FN=2 |
| fix/article | 20 | 16 | FN=4 |
| fix/informal | 15 | 13 | FN=2 |
| fix/homophone | 25 | 18 | FN=7 |
| fix/tense | 25 | 7 | FN=18 |
| preserve/formal | 15 | 0 | FP=0 |
| preserve/clean | 30 | 1 | FP=1 |
| preserve/habitual | 15 | 2 | FP=2 |
| preserve/tech | 30 | 9 | FP=9 |

### Corrector results (on 107 routed sentences)

- Changed 92/95 TP sentences (97% correction rate)
- Corrupted 6/12 FP sentences (50% — sentences that should not have been routed)
- All-correct categories: agreement (18/18), article (16/16), plural (8/8)

### Failure modes identified

**1. fix/tense: 72% miss rate (18/25 FN)**

CoLA judges grammatical structure, not temporal consistency. `"I remove files"` is structurally valid — the model has no concept of `"yesterday"` constraining the tense. Every tense FN scores p > 0.84. This is a structural ceiling: no CoLA classifier can catch temporal inconsistency without understanding the full sentence semantics.

Example FNs:
- `"yesterday I remove all the old log files"` — p=0.926 (not routed)
- `"earlier today I fix the bug and deploy it"` — p=0.925 (not routed)
- `"the team agree to the new approach last week"` — p=0.905 (not routed)

**2. preserve/tech FPs: technical abbreviations score low p(acceptable)**

9 of 12 FPs are technical sentences containing abbreviations. CoLA training data is clean natural language — abbreviated compound tokens look malformed to it.

- `"I updated the Cargo.toml to bump the sherpa dependency."` — p=0.232
- `"The ONNX runtime runs the model in under fifty ms."` — p=0.269
- `"The webhook fires on every merged PR."` — p=0.297
- `"The model runs on ARM64 with NEON optimizations."` — p=0.469

**3. Corrector corrupts technical FPs**

When technical sentences are incorrectly routed, the corrector makes semantic errors on unfamiliar tokens. Trained on CoNLL-14 natural English, it treats abbreviations as errors to fix:

- `"We shard the database by user ID"` → `"We search the database with a user ID"` (meaning corruption)
- `"Cargo.toml to bump the sherpa dependency"` → `"Cargo. Toml to bump into the Sherpa"` (mangled)
- `"INT8 quantized"` → `"INT 8 quantized"` (split token)
- `"ARM64"` → `"ARM 64"` (split token)

The FP+corruption path is worse than doing nothing. A pre-filter that skips the corrector for sentences containing uppercase abbreviations or file-extension patterns could mitigate this. Testing was on developer vocabulary; general-public usage will have different FP patterns (proper nouns, brand names, domain jargon).

### What the pipeline does well

- **ASR duplicates**: 15/15 routed, 14/15 fixed. Near-perfect.
- **Agreement errors**: 18/18 fixed when routed. Perfect.
- **Article errors**: 16/16 fixed when routed. Perfect.
- **Plural errors**: 8/8 fixed when routed. Perfect.
- **Latency**: 9ms clean pass, ~113ms routed (PyTorch CPU; ONNX INT8 on ARM64 estimated ~15ms routed).

### General-public test results

Tested via `scripts/test_pipeline_v3.py`: 270 sentences (155 fix, 115 preserve) using everyday language — personal communication, shopping, family, travel, entertainment. No technical vocabulary.

| Metric | Developer test (v2) | General-public test (v3) |
|--------|--------------------|-----------------------|
| Sentences | 220 | 270 |
| TP | 95 / 130 | 104 / 155 |
| TN | 78 / 90 | 109 / 115 |
| FP | 12 / 90 (13%) | 6 / 115 (5%) |
| FN | 35 / 130 (27%) | 51 / 155 (33%) |
| F1 | 0.80 | 0.78 |
| Corrector corruption | 6 / 12 FP (50%) | 2 / 6 FP (33%) |

**Good news: proper nouns are handled well.** Only 1 FP from 25 proper-noun sentences (Starbucks, Amazon, Disneyland, FaceTime, NYU, etc.). All brand names, person names, and place names scored as acceptable. The technical-abbreviation FP problem from the developer test is largely absent from everyday language.

**Corrector corruption is minimal.** Both FP corruptions were minor preposition changes:
- `"She called to say she will be there by seven."` → changed "will" to "would" (minor tense shift)
- `"I got a gift card to Starbucks for my birthday."` → changed "to" to "from" (preposition preference)

No semantic corruptions of the kind seen in the developer test (ARM64→ARM 64, shard→search).

**New failure mode: fix/contraction has 80% miss rate (12/15 FN).** Contractions without apostrophes (`cant`, `doesnt`, `wont`, `didnt`, `wasnt`, `arent`, `couldnt`, `wouldnt`, `shouldnt`, `hasnt`, `dont`, `hadnt`) score very high p(acceptable) because these are all valid English words. CoLA sees no structural error. The corrector does fix them when routed (3/3 routed contractions were fixed), but the router rarely sends them.

**fix/tense miss rate remains 67% (20/30 FN).** Same structural ceiling as in the developer test.

**Per-category routing in general-public test:**

| Category | Total | Routed | Miss/FP rate |
|----------|-------|--------|-------------|
| fix/asr-dup | 15 | 14 | FN=1 (7%) |
| fix/plural | 10 | 10 | FN=0 (0%) |
| fix/article | 20 | 18 | FN=2 (10%) |
| fix/agreement | 20 | 15 | FN=5 (25%) |
| fix/informal | 15 | 14 | FN=1 (7%) |
| fix/homophone | 30 | 20 | FN=10 (33%) |
| fix/tense | 30 | 10 | FN=20 (67%) |
| fix/contraction | 15 | 3 | FN=12 (80%) |
| preserve/proper-nouns | 25 | 1 | FP=1 (4%) |
| preserve/formal | 15 | 0 | FP=0 (0%) |
| preserve/habitual | 20 | 1 | FP=1 (5%) |
| preserve/work-general | 20 | 1 | FP=1 (5%) |
| preserve/everyday | 35 | 3 | FP=3 (9%) |

**Homophone coverage gap.** The corrector handles some homophones well (their/there, to/too, than/then, passed/past, won/one, whole/hole) but fails on less common ones:
- `"we meat at the coffee shop"` — "meat/meet" not fixed
- `"I left my keys write next to the door"` — "write/right" incorrectly handled (output: "to write next to the door")
- `"I want to no what time"` — "no/know" not fixed (output: "no matter what time")
- `"the store is having a sell"` — "sell/sale" not routed (p=0.924)

**fix/informal quality issues.** The corrector fixes "should of → should have" well but struggles with conditional forms:
- `"we could of left earlier"` → "We could leave earlier" (lost conditional "have")
- `"we could of avoided the whole argument"` → "We could avoid" (lost conditional)
- `"she might of already left by now"` → "She might already be left by now" (grammatically odd)

## Open Questions Before Implementation

1. **Over-correction with the corrector.** The visheratin model modifies ~93% of inputs even when clean. With a good router (DeBERTa CoLA, zero FP) this only runs on flagged sentences, but the corrector itself may still introduce errors on the sentences it does process. Needs testing on the routed subset.

2. **ONNX Runtime on Android.** The app uses sherpa-onnx rather than the `ort` crate directly. DeBERTa and t5-efficient-tiny would require the `ort` crate and a pre-built `libonnxruntime.so` for Android arm64. This is a meaningful build infrastructure change — adds a second native library and complicates the APK.

3. **DeBERTa size on Android.** FP32 is ~283MB. INT8 via `quantize_dynamic` would be ~70MB. Even at 70MB this is the largest single inference asset in the app (current total APK ~51MB). Needs evaluation against acceptable app size increase.

4. **Homophone coverage ceiling.** The 5 FN cases from the CoLA router are structural false negatives — no acceptability classifier can catch `your going to` without a lexical/semantic layer. The corrector fixes some of these when routed to it, but the router will never send them. A targeted rule layer (Harper custom rules for homophone pairs) could complement the neural router.

---

## Recommended Architecture

Based on all empirical testing, the recommended pipeline is:

```
Transcription output
        │
        ▼
electra-small-CoLA  (~4ms, single forward pass)
        │
        ├── P(acceptable) ≥ 0.80 ──► pass through unchanged  (~4ms total)
        │
        └── P(acceptable) < 0.80 ──► route to corrector
                                            │
                                            ▼
                               t5-efficient-tiny  (~50ms ONNX INT8)
                                            │
                                            ▼
                                      Final output  (~55ms total)
```

Router: `pszemraj/electra-small-discriminator-CoLA` at threshold=0.75
- 13.5M params, Apache 2.0, 4ms CPU (estimated 2-5ms ONNX INT8 on ARM64)
- TP=10 TN=8 FP=0 FN=5 on test set — zero false positives on technical text
- Same accuracy as deberta-v3-xsmall at 6x lower latency and 5x smaller
- 0.75 threshold sits in the center of the natural gap [0.684, 0.803] for maximum margin

Corrector: `visheratin/t5-efficient-tiny` (~44MB INT8, ONNX pre-published)
- Pre-published quantized ONNX weights — no export step required
- 4/8 over-corrections vs 3/8 for gec-t5_small — one extra over-correction not worth 99MB
- MIT license
- If tiny proves insufficient after real-world testing, `Unbabel/gec-t5_small` is the upgrade path (~143MB INT8, ONNX export confirmed via `scripts/export_gec_t5_onnx.py` with Python 3.14 patch)
- t5-efficient-mini is not recommended — 6/8 over-corrections, actively worse than tiny on clean text

DistilGPT2 perplexity scoring is not recommended as a router. It cannot distinguish rare technical vocabulary from genuine errors, resulting in TN=0 at any usable threshold.

### Implementation blockers

1. **ORT Android build.** Both models require the `ort` crate and a pre-built `libonnxruntime.so` for Android arm64. The app currently uses sherpa-onnx which bundles its own ORT internally — extracting that for general use or adding a second ORT instance is non-trivial.

2. **gec-t5_small ONNX export.** Resolved — `scripts/export_gec_t5_onnx.py` exports and quantizes to INT8 (~143MB total). Quality verified 8/8. Requires the Python 3.14 descriptor binding patch at the top of the script (Optimum 2.1.0 bug).

3. **DeBERTa/electra model size.** electra-small FP32 is ~54MB. Dynamic INT8 quantization via ORT `quantize_dynamic` should give ~14MB. Acceptable as a model asset.

4. **Homophone coverage ceiling.** The 5 FN cases are a structural ceiling — no CoLA classifier can route `your going to` since it parses syntactically. Addressing these requires a separate lexical layer (Harper homophone rules) that runs independently of the router.

---

## April 2026 follow-up: production observations and next steps

### Router mismatch in production

Once deployed (electra-small router + t5-efficient-tiny, threshold 0.75), real dictation revealed a consistent over-triggering pattern. Conversational sentences like `"Yeah that makes sense. That's fine."` scored below the threshold because "Yeah" at the start registers as informal. The corrector then hallucinated content — `"That's fine."` became `"That makes sense. That is fine."` — a phrase duplication that changed the meaning.

This confirms the structural issue flagged in the v3 test results: the CoLA framing measures formal written acceptability, not transcription correctness. Informal conversational register reliably fails it regardless of whether the ASR output was correct.

The grammar router score is now surfaced in the history detail view per-entry so the threshold can be tuned empirically on real dictation data.

### Short-text gate

Replacement dictation (user highlights 2-3 words and re-dictates) sends fragments through the pipeline. A 2-word fragment like "main branch" scores very low on CoLA (it is not a sentence) and gets rewritten by T5 into something different. A word-count gate (`MIN_GRAMMAR_WORDS = 5`) skips the grammar stage entirely for short inputs. At or below 5 words there is not enough context for either the router or corrector to make reliable decisions.

### Is CoLA the right framing? No.

The router should answer "does this text contain a transcription error?" not "is this grammatically acceptable formal English?". These are different questions. Conversational speech fails CoLA even when perfect. Subtle ASR word substitutions can pass it.

The most promising alternative reuses the same ELECTRA-small architecture already embedded, but with a different task head. `google/electra-small-discriminator` was pretrained to predict which tokens were "replaced" by a generator (the ELECTRA training objective). This maps directly to detecting substituted ASR words. `Xenova/electra-small-discriminator` has a 14MB INT8 ONNX export on HuggingFace. The routing signal would be max per-token replaced-probability rather than a CLS-level sentence score — informal register would not affect it, but phonetically-substituted words would score high.

### Alternative correctors evaluated (April 2026)

The following correctors now have known ONNX-ready exports. Key finding: models trained on synthetic typo data (visheratin family, C4_200M) are more prone to hallucination than models trained on annotated GEC corpora or real learner errors.

| Model | ONNX ready | INT8 total | Training data | Notes |
|---|---|---|---|---|
| visheratin/t5-efficient-tiny (current) | Yes | ~32 MB | C4_200M synthetic | Hallucination risk on informal text |
| visheratin/t5-efficient-mini | Yes | ~56 MB | C4_200M synthetic | 6/8 over-corrections — worse than tiny |
| onnx-community/grammar-synthesis-small-ONNX | Yes | ~91 MB | JFLEG real errors | Stated goal: preserve correct text. But earlier v2 tests show JFLEG models make semantic changes — see eliminated candidates above |
| JonaWhisper/jonawhisper-gec-t5-small-onnx | Yes | ~94 MB | cLang-8 via Unbabel | Unbabel/gec-t5_small export — confirmed working, best quality at 3/8 over-corrections |
| gotutiyan/gec-t5-small-clang8 (self-convert) | No | ~77 MB est. | cLang-8 | No ONNX export, requires manual conversion |
| GECToR roberta-base (self-convert) | No | ~130 MB | BEA19+W&I | Fixed edit vocabulary — cannot hallucinate new text. Best architecture for safety |

**Note on grammar-synthesis-small (JFLEG):** Earlier corrector testing showed JFLEG-trained models rewrite fluency rather than make minimal corrections — `"The quick brown fox"` became `"The yellow fox"`, `"API endpoints"` became `"API ends"`. The `onnx-community` export may behave differently if based on a different checkpoint, but treat as unverified until tested against the v2/v3 test sets.

### GECToR: architecture-level fix for hallucination

GECToR (Grammatical Error Correction as Sequence Tagging) outputs per-token edit operations from a fixed vocabulary of ~5000 transformations (KEEP, DELETE, verb inflections, article changes, preposition swaps). It cannot produce arbitrary text — every output token is one of the predefined edit operations. This eliminates hallucination at the architecture level.

Available models all lack ONNX exports and require manual safetensors → ONNX conversion. INT8 sizes: roberta-base ~130MB, deberta-base ~125MB. No small-backbone variant exists. Worth the conversion effort if hallucination remains a problem after router improvements.

### Recommended next step

Switch the router from CoLA sentence scoring to ELECTRA discriminator per-token scoring. Same 14MB model family, no file size increase, eliminates the informal-register false-trigger problem. Implement `route_per_token()` that runs `google/electra-small-discriminator` (not the CoLA fine-tune) and returns the max per-token replaced-probability as the routing score. File naming in `data/grammar/` stays unchanged since the file format is the same.
