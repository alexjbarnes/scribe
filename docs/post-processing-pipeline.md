# Post-Processing Pipeline

The post-processing pipeline runs on the full joined transcript (all VAD chunks concatenated) after transcription completes. It lives in `src-tauri/src/postprocess/`.

Pipeline order: Raw transcription -> Filler removal -> ITN -> Harper -> Spelling -> Cleanup

Each stage receives the output of the previous stage. All stages are recorded for debugging (visible under "Details" in the history UI).

## Stage 1: Filler Removal (`filler.rs`)

Strips spoken disfluencies. Runs in two passes:

**Pass 1 -- Multi-word phrases** (case-insensitive, word-boundary-aware):
- "you know"
- "i mean"
- "sort of"
- "kind of"
- "you see"
- "i guess"
- "i think um"
- "i think uh"

**Pass 2 -- Single words** (case-insensitive, after punctuation stripping):
- um, umm, uh, uhh, er, err, ah, ahh, hmm, hm, mm, mmm, mhm

**Pass 3 -- Consecutive duplicate words** (e.g. "I I think" becomes "I think").

### What you can tweak

- **Add filler phrases**: Edit `FILLER_PHRASES` in `filler.rs`. Phrases are removed before single words, so multi-word patterns take priority. Word-boundary matching prevents false positives (e.g. "you know" inside "do you know" will still match since it checks whitespace boundaries, not semantic boundaries).
- **Add filler words**: Edit `FILLER_WORDS` in `filler.rs`. Only standalone tokens are removed. A word like "ah" won't be stripped from "ahoy" because the tokenizer splits on whitespace first.
- **Disable duplicate removal**: The consecutive-duplicate logic has no toggle. To disable it, remove the `if bare == prev_bare` block in `remove_fillers()`.
- **Risk**: Aggressive filler lists can eat legitimate words. "I mean" as a filler phrase will remove it even when used literally ("I mean the river" becomes "the river"). Test with real transcripts before adding entries.

## Stage 2: Inverse Text Normalization (`itn.rs`)

Converts spoken number/date/abbreviation forms to written equivalents.

### Number handling

Cardinal words are mapped via `CARDINALS` (zero through ninety) and combined with `MULTIPLIERS` (hundred, thousand, million, billion). The parser handles compound forms:
- "twenty three" -> "23"
- "one hundred twenty three" -> "123"
- "two thousand twenty six" -> "2026"
- "one million" -> "1000000"
- "one hundred and twenty" -> "120" (the word "and" is skipped between number words)

### Currency and percent

When a number sequence is followed by "dollars"/"dollar", it becomes `$N`. When followed by "percent", it becomes `N%`.
- "twenty three dollars" -> "$23"
- "fifty percent" -> "50%"

### Dates

Month names (`MONTHS` map, all 12) followed by ordinals become "Month N":
- "january fifth" -> "January 5"

### Ordinals

Standalone ordinal words (`ORDINALS` map, first through twentieth plus thirtieth) convert to numeric form:
- "third" -> "3rd"

### Abbreviations

Single-word and multi-word abbreviations (`ABBREVIATIONS` map):
- mister -> Mr.
- missus -> Mrs.
- miss -> Ms.
- doctor -> Dr.
- professor -> Prof.
- versus -> vs.
- etcetera / et cetera -> etc.

### What you can tweak

- **Extend number range**: `CARDINALS` only goes to 90. To handle "one hundred" style constructions the multiplier system works, but standalone words like "ninety one" require adding entries up to 99, or changing the parser to combine tens + units implicitly.
- **Add currencies**: Only USD ("dollars") is supported. Add new currency suffixes in the `match next_bare` block inside `normalize()`. E.g. add "pounds" -> prepend with a pound sign.
- **Add abbreviations**: Edit the `ABBREVIATIONS` map. Multi-word abbreviations need separate handling in the `replace_abbreviation` call at the top of `normalize()` (see "et cetera" as an example).
- **Ordinal gaps**: `ORDINALS` maps first through twentieth plus thirtieth. Twenty-first through twenty-ninth are missing, as are all ordinals above thirtieth. Add them to the `ORDINALS` map if needed.
- **Decimal numbers**: Not supported. "three point five" stays as words. Would require extending `try_parse_number_sequence` to handle a "point" token.
- **Time expressions**: Not supported. "three thirty pm" stays as words. Would require a dedicated time parser.

## Stage 3: Harper Grammar Polish (`harper.rs`)

Uses harper-core (pure Rust, no network calls) with ~200 curated grammar rules. Runs in ~5-10ms.

Configuration:
- **Dictionary**: `FstDictionary::curated()` -- Harper's built-in FST dictionary
- **Dialect**: `Dialect::American`
- **Rule set**: `LintGroup::new_curated` -- the full curated lint set

Harper finds lints (grammar/spelling issues) and applies the first suggestion for each lint. Fixes are applied from end-to-start to avoid offset invalidation.

### What you can tweak

- **Dialect**: Change `Dialect::American` to `Dialect::British` (or other variants if Harper adds them) in `HarperLinter::new()`.
- **Disable specific rules**: `LintGroup` supports `config_mut()` to disable individual rules. Currently all curated rules are enabled. To disable a rule, call something like `linter.config_mut().set_rule("RuleName", false)` after construction. Check harper-core docs for available rule names.
- **Custom dictionary**: Replace `FstDictionary::curated()` with a custom dictionary to add domain-specific words that Harper might flag as misspellings.
- **Suggestion strategy**: Currently always applies `suggestions[0]` (first suggestion). You could filter by lint severity or skip certain lint categories.

## Stage 4: SymSpell Spell Correction (`spelling.rs`)

Uses the `symspell` crate (edit-distance-based correction) with English frequency dictionaries. Catches ASR transcription misspellings that Harper's dictionary misses. Sub-millisecond per sentence.

Dictionaries are embedded at compile time via `include_str!`:
- **Unigram**: 82,765 English words with frequency counts (~1.8MB)
- **Bigram**: 243,342 word pairs with frequency counts (~4.4MB)

Uses `lookup_compound()` for sentence-level correction (handles split/joined words) and falls back to per-word `lookup()` when the sentence contains protected words.

### Protected words

The spell corrector preserves words that should not be corrected:
- **All-uppercase** (2+ chars): acronyms like API, DNS, POC
- **Contains digits**: numbers, version strings (v2, 3rd)
- **Starts uppercase mid-sentence**: likely proper nouns (heuristic)

### What you can tweak

- **Edit distance**: `max_dictionary_edit_distance(2)` in `SpellCorrector::new()`. Lower to 1 for fewer false corrections (but misses more errors). Higher than 2 is not recommended (exponential candidate explosion).
- **Custom dictionary terms**: Add domain-specific words to the unigram dictionary file (`data/frequency_dictionary_en_82_765.txt`) with a high frequency count to prevent them being "corrected" away.
- **Protection rules**: Edit `should_protect()` in `spelling.rs` to adjust which words are exempt from correction.

## Stage 5: Final Cleanup (`mod.rs`)

Deterministic text normalization applied last.

1. **Trim** leading/trailing whitespace
2. **Collapse multiple spaces** into single spaces
3. **Capitalize first character** (ASCII lowercase only)
4. **Add trailing period** if the last character is not one of: `.` `!` `?` `,` `;` `:` `"` `'` `)`

### What you can tweak

- **Sentence-ending punctuation set**: The `matches!` guard in `final_cleanup()` controls which characters suppress the auto-period. Add or remove characters from that list.
- **Capitalization**: Currently only handles ASCII lowercase first characters. Unicode capitalization would require `.to_uppercase()` instead of `.to_ascii_uppercase()`.
- **Auto-period behavior**: Some use cases (e.g. filling in form fields, chat messages) may not want a trailing period. This would need a config flag or a way to skip the cleanup stage.

## Pipeline-level configuration

- **Stage ordering**: Defined in `postprocess()` in `mod.rs`. Changing the order matters. Filler removal before ITN prevents "um twenty three" from partially normalizing. Harper after ITN means grammar rules see "$23" not "twenty three dollars". Spelling after Harper means it only sees text that Harper has already cleaned up, avoiding duplicate corrections.
- **Skipping stages**: No runtime toggle exists. To skip a stage, comment it out in `postprocess()` or add a config check. Each stage is independent -- removing one won't break others.
- **Adding stages**: Add a new module, call it from `postprocess()` between existing stages, and push a `PipelineStage` with the result. The UI will show it automatically.

## Performance

Typical latency for the full pipeline on a sentence:
- Filler removal: ~1ms
- ITN: ~5ms
- Harper: ~5-10ms
- Spelling: <1ms
- Cleanup: <1ms

Total: ~10-15ms. All stages run synchronously on the calling thread. Per-stage and total timings are recorded in history entries.

## Future: neural grammar correction (GECToR)

Evaluated March 2026. The rule-based pipeline (Harper) handles spelling, punctuation, and simple grammar but misses context-dependent errors like verb tense ("we remove" should be "we removed" when narrating past events).

### What was tested

GECToR (Grammatical Error Correction using Token tagging) is a single-forward-pass token classification model. Unlike seq2seq models (T5, etc.) that decode one token at a time, GECToR predicts a correction tag per input token in one pass, then optionally iterates. Tags include verb tense transformations, noun number changes, insertions, and deletions.

Tested `gotutiyan/gector-roberta-base-5k` (128M params) and `gotutiyan/gector-bert-base-cased-5k` (112M params) on real ASR problem sentences.

### Results

Correctly fixed:
- "Yesterday I remove all the old code" -> "Yesterday I removed all the old code"
- "Last week we remove the database and start fresh" -> "removed...started" (both verbs)
- "I already remove the file" -> "I have already removed the file" (added auxiliary)
- "Earlier today I fix the bug and deploy it" -> "fixed...deployed"
- "The pipeline remove fillers" -> "The pipeline removes fillers" (subject-verb agreement)
- Run-on sentences got punctuation and capitalization added

Correctly left alone:
- "We remove old files every Friday" (present tense habitual, correct)

Did not fix:
- "So we remove the views and everything" -- the original problem sentence. No temporal marker means no signal for past tense. This is a discourse-level inference problem.

False corrections observed:
- BERT-base: "I always fix bugs before deploying" -> "deploying. them" (inserted garbage)
- BERT-base: doubled periods in long run-on sentences
- RoBERTa-base was cleaner overall, fewer false corrections

### Latency (PyTorch CPU, no ONNX, no quantization)

Single sentence:
- No correction needed: ~75ms (1 iteration)
- Correction needed: ~145-165ms (2 iterations)
- Complex sentence: ~300ms (4 iterations)

Batched (12 sentences):
- 1 iteration: ~59ms/sentence
- 3 iterations: ~95ms/sentence

ONNX + INT8 quantization would roughly halve these numbers.

### Size

- RoBERTa-base: 128M params, ~490MB FP32, ~120MB INT8 estimated
- BERT-base-cased: 112M params, ~426MB FP32, ~106MB INT8 estimated
- No DistilBERT GECToR variant exists (would need custom training)

### Assessment

Pros:
- Fixes real errors that Harper cannot (tense with temporal markers, subject-verb agreement)
- Single forward pass architecture keeps latency bounded (unlike seq2seq)
- Confidence threshold can filter low-confidence corrections to reduce false positives

Cons:
- 100-120MB INT8 is a large addition to the app (current APK is ~51MB)
- ~30-40ms estimated ONNX INT8 latency doubles the pipeline time
- Cannot fix the hardest case (tense from discourse context without temporal markers)
- Occasionally introduces false corrections (needs careful threshold tuning)
- Implementation requires ~500 lines of Rust: WordPiece tokenizer, ONNX inference, tag application logic
- Pre-trained weights are non-commercial license; commercial use requires training your own

### If revisiting

- Check if gotutiyan publishes a DistilBERT (66M param) variant, which would halve size and latency
- Consider training a custom GECToR on DistilBERT with the MIT-licensed training code
- The Unbabel/gec-t5_small (60M params, Apache-2.0) is permissively licensed but seq2seq, so too slow
- Microsoft's EdgeFormer (11M params) is the right architecture but weights are not public

## Future: broader grammar/correction options

Researched March 2026. Survey of all approaches for improving post-processing beyond the current Harper rule-based system.

### Option comparison

| Approach | Size | Latency (CPU) | Quality | Effort |
|----------|------|---------------|---------|--------|
| Harper (current) | 0 (pure Rust) | 5-10ms | ~200 rules | Already in use |
| Custom Harper rules for ASR | 0 | <1ms | Targeted | Low |
| SymSpell spell correction | ~5MB dict | <1ms | Spell-only | Low |
| sherpa-onnx punctuation model | 7.1MB INT8 | ~10-20ms | Good punct/case | Low-Medium |
| EdgeFormer (Microsoft) | ~11MB INT8 | <100ms | F0.5=52.7 CoNLL-14 | High |
| GECToR + MobileBERT | ~7MB INT8 | ~40-80ms est. | Unknown (no pretrained) | Very High |
| GECToR + RoBERTa-base | ~120MB INT8 | ~75-160ms | High | Medium |
| T5-small grammar | ~140MB INT8 | 100-300ms | Moderate | Medium |

### Phase 1: rule-based expansion (no models)

**Custom Harper rules for ASR artifacts.** Harper's `Linter` trait allows custom rule structs. Target patterns:
- Homophones ("there/their/they're", "its/it's") using surrounding word context
- Common ASR contractions ("wanna" -> "want to", "gonna" -> "going to")
- Number/word boundary issues
- Missing apostrophes in contractions

Limitation: rule-based hits a ceiling. Catches known patterns but misses novel errors.

**SymSpell for spell correction.** The `fast_symspell` Rust crate provides edit-distance-1/2 correction in microseconds using a frequency dictionary. Handles ASR transcription typos that Harper's limited dictionary might miss. Dictionary can be customized with domain terms. No contextual awareness though ("there" vs "their" requires context).

### Phase 2: sherpa-onnx punctuation model

sherpa-onnx ships an English punctuation/truecasing model:
- INT8: **7.1MB**, FP32: 28MB
- Already a dependency, Rust bindings exist
- Adds punctuation and truecasing to raw ASR output

Primary value: fixing punctuation at chunk boundaries when joining VAD segments. Marginal value if the ASR model already handles punctuation well (Whisper does, Parakeet less so).

Available at: https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models

### Phase 3: neural grammar correction

**EdgeFormer (most promising under 50MB).** Microsoft's on-device grammar model, powers Microsoft Editor:
- 11M params (12 encoder layers, 2 decoder layers, 512 hidden, 8 heads)
- ~11MB on disk INT8, 42MB peak RAM
- <100ms latency per sentence
- Open source checkpoint at microsoft/unilm/edgelm

The catch: ONNX export from the public fairseq checkpoint is poorly documented. No evidence of anyone running it on Android ARM64. Would integrate via the `ort` crate (pykeio/ort, wraps ONNX Runtime).

**GECToR with smaller backbone.** The GECToR training code is Apache 2.0. The tagging architecture supports any encoder backbone:
- MobileBERT: ~25M params, ~7MB INT8 (would need custom training)
- DistilBERT: ~66M params, ~17MB INT8 (would need custom training)
- No off-the-shelf small GECToR exists. Grammarly experimented with distillation but did not release smaller models.

**T5-small.** Total INT8 is ~140MB (encoder 35MB + decoder 108MB). Too large and autoregressive decoding is too slow (each output token requires a full decoder pass).

### What Grammarly and Microsoft ship

**Grammarly (2025):** T5-based model, <300MB memory, 297 tokens/sec on Apple M-series via MLX. Newer approach uses ~1B Llama variant at 210 tok/s. Both well beyond mobile size constraints.

**Microsoft Editor (2022-present):** EdgeFormer at 11M params, <50MB RAM, <100ms, ONNX Runtime. Closest to feasible for Scribe. Ships encoder+decoder+beam search as a single ONNX graph.

### Assessment

Rule-based expansion (Harper + SymSpell) gets 80% of the value at 0% model cost. The sherpa-onnx punctuation model is a cheap add if chunk-boundary punctuation is a problem.

For neural: EdgeFormer is the right target if the ONNX export can be made to work. Training a small GECToR is the fallback. Both require ML expertise and carry risk. T5 and LLM-based approaches are too large for mobile.
