# Mid-sentence Capitalisation Fix — Known Issues

## Background

`fix_mid_sentence_caps()` in `src-tauri/src/postprocess/mod.rs` was added to
handle cases where the Parakeet model spuriously capitalises words mid-sentence
(e.g. "What We Want" instead of "What we want"). It runs as part of the final
cleanup stage, after grammar and spell correction.

## Known failures

### 1. "I" contractions lowercased

Affected words: I've, I'd, I'll, I'm, I've

The function determines the "core" of a word by scanning to the first character
that is not alphabetic and not an apostrophe:

```rust
let alpha_end = rest.find(|c: char| !c.is_alphabetic() && c != '\'').unwrap_or(rest.len());
let core = &rest[..alpha_end];  // "I've" → core = "I've", NOT "I"
```

The standalone-I guard (`core == "I"`) never fires because `core` is "I've",
not "I". The all-uppercase acronym check also fails (v, e are lowercase).
Result: "I've" → "i've".

### 2. Mixed alphanumeric tokens lowercased

Affected words: R2, v3, C4, H2O, INT8, etc.

The `alpha_end` scan stops at the digit, giving `core = "R"` (length 1).
The acronym guard requires `core.len() >= 2`, so it doesn't fire.
The first letter is lowercased, the digit is unchanged:
"R2" → "r2", "v3" → "v3" (already lowercase so unaffected), "C4" → "c4".

### 3. Regression against the grammar stage

The grammar stage (nlprule) runs *before* cleanup and already handles
capitalisation rules correctly. In the trace below, the grammar stage
produces the right output, then cleanup breaks it:

```
Grammar:  ...once I've settled the R2 bucket?
Cleanup:  ...once i've settled the r2 bucket?
```

`fix_mid_sentence_caps` is fighting the grammar checker's correct output.

## Proposed fixes

### Short-term: patch the two specific gaps

**Fix 1** — preserve "I" contractions: after extracting `core`, also check
whether the word starts with `"I'"` (capital I followed immediately by
apostrophe):

```rust
// Preserve "I" and "I've", "I'd", "I'll", "I'm" etc.
if core == "I" || rest.starts_with("I'") {
    result.push(word.to_string());
    continue;
}
```

**Fix 2** — preserve mixed alphanumeric tokens: if the word (after stripping
leading punctuation) contains any digit, leave it untouched:

```rust
// Preserve tokens containing digits: R2, v3, INT8, etc.
if rest.chars().any(|c| c.is_ascii_digit()) {
    result.push(word.to_string());
    continue;
}
```

Both checks should be added before the acronym check.

### Medium-term: reconsider the approach

`fix_mid_sentence_caps` is a blunt heuristic. The problem it solves (spurious
mid-sentence caps from the model) is real, but new failure cases will surface
as the app is used with more varied vocabulary.

Options worth evaluating:

- **Trust the grammar stage** and remove `fix_mid_sentence_caps` entirely.
  Risk: nlprule does not flag spurious capitalisation as a grammar error, so
  Parakeet's false caps ("What We Want") would pass through uncorrected.

- **Run fix_mid_sentence_caps before the grammar stage** — this does NOT help.
  nlprule is a grammar rule engine, not a proper noun or identifier database.
  It would not re-capitalise tokens lowercased by our pass (e.g. "r2" would
  stay "r2"). The grammar stage only preserved "R2" in the trace because it
  had no rule to change it — not because it understood the token.

- **Expand the preserve list** — continue patching specific categories as
  failures are observed (e.g. proper nouns that consistently appear in
  dictation, version strings, product names).

## Test cases to add

```rust
fix_mid_sentence_caps("once I've settled the R2 bucket")
  → "once I've settled the R2 bucket"  // currently fails

fix_mid_sentence_caps("connect to the C4 instance via v3")
  → "connect to the C4 instance via v3"  // currently "c4", v3 already ok

fix_mid_sentence_caps("then I'd like to check INT8 performance")
  → "then I'd like to check INT8 performance"  // I'd fails, INT8 ok (all-caps)
```
