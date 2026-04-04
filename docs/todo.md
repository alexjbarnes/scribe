# Todo

## Grammar correction

- [x] Selective negation revert: replace all-or-nothing negation guard with word-level diff that reverts only negation-changing edits while keeping other corrections
- [x] KV cache for T5 decoder: use decoder_with_past model to avoid re-running full decoder each step, roughly 2-3x speedup
- [ ] Fine-tune corrector on transcription-specific error pairs (need 60+ examples in scripts/data/corrector_test_cases.json)
- [ ] Fine-tune router on acceptability labels (need 50+ examples in scripts/data/router_test_cases.json)
- [ ] Remove nlprule fallback (replaced by neural correction)

## Snippets

- [ ] Phonetic matching for snippet triggers (Metaphone) to handle ASR substitutions like "female" vs "email"
