# Grammar Correction Roadmap

## Current state

Post-processing uses Harper (~300 curated Rust rules) and SymSpell for spell correction. Harper catches basic grammar and punctuation issues but misses common transcription errors:

- a/an agreement ("a error" should be "an error")
- Word transpositions ("tot he" should be "to the")
- Subject-verb disagreement ("he walk" should be "he walks")
- Mid-sentence capitalization from ASR chunk boundaries
- Context-dependent errors (verb tense, homophones)

LanguageTool covers all of these with ~3000 English rules, but requires Java.

## Plan: Replace Harper with nlprule

nlprule is a Rust port of LanguageTool's rule engine. It compiles LanguageTool's XML rule definitions into binary data at build time and interprets them at runtime. Pure Rust, no JVM, runs in-process at 1.7-2.8x the speed of LanguageTool.

The upstream repo (github.com/bminixhofer/nlprule) is unmaintained since 2021. We are forking it and fixing the unimplemented features that prevent ~600 rules from compiling.

### Fork work phases

**Phase 1: Structural parsing fixes (unlocks ~230 rules)**

Low-risk serde changes. Add missing optional fields to rule/rulegroup structs so the XML deserializer doesn't reject rules that use them.

- Add `tags` field to rule and rulegroup structs (parse and ignore) -- 21 rules
- Add `type` field to rule and rulegroup structs -- 15 rules
- Handle `example` elements with `type` attribute -- 43 rules
- Support detection-only rules (no suggestion, just flag the error) -- 35 rules
- Fix 3 regex syntax differences between Java and Rust regex engines
- Handle `raw_pos`, missing `$value`, and other minor structural gaps

**Phase 2: Match postag in suggestions (unlocks ~242 rules)**

The highest-value single change. LanguageTool suggestions can reference POS tags to inflect words correctly. For example, if "he walk" matches and the suggestion says `postag="VBZ"`, the engine looks up the VBZ form of "walk" and produces "walks".

- Look up alternative word forms from the tagger dictionary
- Filter by POS tag or POS tag regex
- Handle `postag_replace` (regex replacement on POS tag strings)
- Handle `text` attribute in match elements

This unlocks a/an agreement, subject-verb agreement, and many other rules that need morphological awareness.

**Phase 3: Filters (unlocks ~31 rules)**

LanguageTool filters are Java callbacks for custom logic (date validation, number checking, etc.). Reimplement the most common ones in Rust. Start with the filters that cover the most rules.

**Phase 4: Update rule data**

- Build tokenizer and rules from latest LanguageTool XML data (currently based on v5.2)
- Investigate OpenNLP model format changes for the tokenizer model
- Update the binary data generation pipeline

### Size

| Component | Compressed | On disk |
|-----------|-----------|---------|
| en_rules.bin | 0.96 MB | 7.2 MB |
| en_tokenizer.bin | 6.8 MB | 11.1 MB |
| Data files total | 7.8 MB | 18.3 MB |
| Compiled crate (stripped, LTO) | -- | 1.2 MB |
| Grand total | -- | ~19.5 MB |

Data files are derived from LanguageTool and licensed LGPL v2.1. The crate itself is MIT/Apache-2.0.

### Delivery

Embed the rule data in the app binary (compile-time inclusion). 7.8 MB compressed is acceptable given we already ship 80MB+ of ONNX models. No runtime downloads for grammar rules.

## User-defined custom rules

Users need a way to add domain-specific corrections from the app UI without touching XML or recompiling anything.

### Approach: Simple JSON rules loaded at runtime

A lightweight format for word/phrase substitutions that covers 90% of user needs:

```json
[
  {
    "id": "gonna_going_to",
    "match": "gonna",
    "replace": "going to",
    "message": "Did you mean going to?"
  },
  {
    "id": "kubernetes_k8s",
    "match": "kubernetes",
    "replace": "Kubernetes",
    "message": "Capitalize Kubernetes"
  },
  {
    "id": "wanna_want_to",
    "match": "wanna",
    "replace": "want to"
  }
]
```

Fields:
- `id` -- unique identifier, auto-generated if not provided
- `match` -- word or phrase to match (case-insensitive by default)
- `replace` -- replacement text
- `message` -- optional explanation shown in history pipeline details
- `case_sensitive` -- optional, defaults to false
- `enabled` -- optional, defaults to true

### How it works

1. User opens Settings > Custom Rules in the app
2. Adds a trigger phrase and replacement
3. Rule saved to `custom_rules.json` in the app data directory
4. On next transcription, the custom rules run as a pipeline stage after nlprule and before final cleanup
5. Rules are loaded once at startup and reloaded when the file changes

### Implementation

Custom rules translate directly into nlprule's internal `Rule` structs at load time. No XML compilation needed. The matching is simple string/phrase matching against tokenized text, which nlprule's pattern engine already supports.

The pipeline stage order becomes:

1. Filler removal (rule-based)
2. Inverse text normalization (numbers, dates, ordinals)
3. nlprule grammar correction (~3000 LanguageTool rules)
4. User custom rules (JSON phrase substitutions)
5. Spell correction (SymSpell)
6. Final cleanup (capitalize, trailing punctuation)

User rules run after nlprule so they can override or supplement LanguageTool corrections. Spell correction runs last so it doesn't "correct" intentional user replacements.

### UI

The custom rules UI is a simple list with add/edit/delete:

- Text field for trigger phrase
- Text field for replacement
- Toggle to enable/disable each rule
- Rules display in a scrollable list
- Import/export as JSON for sharing between devices

Same UI on all platforms (part of the Settings tab in the Tauri WebView).

### What this does not cover

User rules handle word/phrase substitutions only. They do not support:
- POS-tag-aware matching (use nlprule's XML rules for that)
- Regex patterns (keep it simple for non-technical users)
- Context-dependent corrections (homophones, verb tense)

These require the full nlprule engine. If a user needs something more advanced, they can contribute an XML rule to the base ruleset.

## Migration plan

1. Fork nlprule, complete Phase 1 and Phase 2
2. Add nlprule as a pipeline stage alongside Harper (both running, compare output)
3. Validate that nlprule catches everything Harper catches plus more
4. Remove Harper dependency
5. Add user custom rules UI
6. Complete Phase 3 and Phase 4 as needed

## Constraints

- Pure Rust, no JVM dependency
- Must work on Android arm64 (no dynamic linking to system libs)
- Runtime budget: under 10ms for a typical dictation sentence
- No network requests for grammar rules (data compiled in or loaded from local files)
- English only for now, architecture stays language-agnostic
