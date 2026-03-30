# Grammar Correction Roadmap

Current post-processing uses Harper's ~200 curated rules (see [post-processing-pipeline.md](post-processing-pipeline.md) for details). This document covers the plan for improving grammar correction beyond what Harper provides out of the box.

## Problem

Harper catches spelling, punctuation, and simple grammar issues but misses common transcription errors:
- Mid-sentence capitalization from ASR ("why Does this happen" — "Does" should be "does")
- ASR-specific misspellings not in Harper's dictionary
- Context-dependent errors (verb tense, homophones) that require more than pattern matching

## Planned Architecture

| Layer | Tool | Latency | Updateable | Status |
|-------|------|---------|------------|--------|
| 1. Rules | Harper curated + base Weirpack + user Weirpack | ~5-10ms | Yes (ZIP files via CDN) | Harper curated in use; Weirpacks not yet implemented |
| 2. Capitalization | Custom `Linter` trait impl (mid-sentence caps fix) | <1ms | Compiled | Not yet implemented |
| 3. Spell | SymSpell (`fast_symspell` crate) | <1ms | Dictionary file | Not yet implemented |
| 4. ML GEC (optional) | GECToR INT8 ONNX via `ort` | 40-80ms CPU | Model file | Evaluated, not integrated |

Layers 1-3 are immediate priorities. Layer 4 is optional/toggleable and longer-term.

## Layer 1: Harper Weir DSL + Weirpacks

### What Weir is

Harper supports custom grammar rules via the **Weir DSL**, a pattern-matching language for word sequences. Rules are packaged as **Weirpacks** — ZIP archives of `.weir` files that load at runtime.

### Weir syntax

```weir
expr main (gong to)
let becomes "going to"
let message "Did you mean `going to`?"
let kind "Typo"
test "I am gong to go." "I am going to go."
```

Rules match word patterns and suggest replacements. Each rule includes a test case.

### Runtime loading

Weirpacks load from disk via `Weirpack::from_bytes()` — they are not compiled into the binary. This enables:

- **Base ruleset**: Ship a default Weirpack with ASR-specific corrections (homophones, common mistranscriptions). Push updates via CDN/manifest without app updates.
- **User-defined rules**: Users create their own `.weir` files in-app for domain-specific corrections.
- **Merging**: `LintGroup::merge_from()` supports layering multiple packs. Name collisions resolve by last-pack-wins (intentional override). This means we can ship a base pack and let users append their own on top.

### Limitations

Weir matches **word patterns**, not arbitrary text transformations. Mid-sentence capitalization ("why Does") cannot be fixed with a Weir rule because it requires checking casing rather than matching specific words. This needs a custom `Linter` trait implementation (Layer 2).

### Delivery model

```
CDN manifest.json
├── version: "1.2.0"
├── base_weirpack_url: "https://cdn.example.com/rules/base-v1.2.0.weirpack"
└── checksum: "sha256:..."

App startup:
1. Check manifest version against local cache
2. Download new base Weirpack if version changed
3. Load base Weirpack + user Weirpack from disk
4. Merge into LintGroup
```

## Layer 2: Custom Capitalization Linter

A Rust implementation of Harper's `Linter` trait specifically for fixing mid-sentence capitalization from ASR output. Harper's curated rules don't cover this because it's not a standard grammar error — it's an ASR artifact where the model capitalises words at chunk boundaries.

The linter would:
1. Tokenize the sentence
2. For each word that isn't the first word, a proper noun, or an acronym: check if it's capitalised
3. If capitalised and not in a proper-noun dictionary: suggest lowercasing

This is compiled into the binary (not a Weirpack rule) because it requires programmatic logic beyond pattern matching.

## Layer 3: SymSpell Spell Correction

The `fast_symspell` Rust crate provides edit-distance-1/2 correction in microseconds using a frequency dictionary.

- **Size**: ~5MB dictionary file
- **Latency**: sub-millisecond
- **Value**: Catches ASR transcription typos that Harper's limited dictionary misses
- **Limitation**: No contextual awareness ("there" vs "their" requires context, not edit distance)

Would be a new pipeline stage in `src-tauri/src/postprocess/`, inserted between Harper and final cleanup.

The dictionary file can be customised with domain-specific terms and updated independently of the binary.

## Layer 4: ML Grammar Correction (GECToR)

See the detailed evaluation in [post-processing-pipeline.md](post-processing-pipeline.md#future-neural-grammar-correction-gector) and [post-processing-pipeline.md](post-processing-pipeline.md#future-broader-grammarcorrection-options).

Summary: GECToR (tag-based, single forward pass) fixes real errors Harper cannot — verb tense with temporal markers, subject-verb agreement, run-on sentences. Trade-offs are size (~120MB INT8 for RoBERTa-base) and latency (~40-80ms ONNX INT8). Would be an optional/toggleable pipeline stage.

### Alternative: nlprule

Rust port of LanguageTool's rule engine (~85% of English grammar rules). Unmaintained since April 2021 (v0.6.4). 94 transitive crate dependencies. Has a `fancy-regex` feature flag to avoid the C dependency on Oniguruma (`onig_sys`).

- 1.7-2.8x faster than LanguageTool JVM
- Would give LanguageTool-level coverage without JVM dependency
- Rules loaded from binary data files at runtime, not compiled in

### Size

| Component | Compressed (gzip) | On disk |
|-----------|-------------------|---------|
| `en_rules.bin` | 0.96 MB | 7.2 MB |
| `en_tokenizer.bin` | 6.8 MB | 11.1 MB |
| **Data files total** | **7.8 MB** | **18.3 MB** |
| Compiled crate (stripped, LTO) | — | 1.2 MB |
| **Grand total** | — | **~19.5 MB** |

Data files are derived from LanguageTool v5.2 and licensed LGPL v2.1 (the crate itself is MIT/Apache-2.0). The `nlprule-build` companion crate can auto-download the compressed `.bin.gz` files at build time from GitHub releases.

### Delivery model

The data files compress well enough that they don't need to ship in the binary. A reasonable approach:

1. Ship the app without nlprule data files (no size impact on initial download)
2. On first launch (or when user enables "advanced grammar"), download the compressed files (~7.8 MB)
3. Decompress to app data directory (~18.3 MB on disk)
4. Check for updates against a manifest (same mechanism as Weirpack updates)

This keeps the APK/DMG lean and makes nlprule an opt-in enhancement.

### Risk

Stale codebase with no upstream maintenance — last release April 2021, last push May 2023. Would need to fork and own maintenance. 94 crate dependencies is a large surface area. Could be worth it for the breadth of rules if Harper + Weirpacks prove insufficient.

## Priority and Sequencing

1. **Weirpack infrastructure** — Build the loading/merging/caching pipeline. Ship a base Weirpack with common ASR corrections. This is the highest-value work because it makes grammar correction extensible without app updates.
2. **Custom capitalization linter** — Implement the `Linter` trait for mid-sentence caps. Quick win for one of the most visible ASR artifacts.
3. **SymSpell integration** — Add as a pipeline stage. Low effort, immediate value for misspellings.
4. **GECToR / nlprule** — Evaluate when layers 1-3 are in production and we have data on what errors remain. Check for smaller GECToR variants (DistilBERT, MobileBERT) or a maintained nlprule fork before committing.
