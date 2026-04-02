# User Vocabulary

User vocab is a runtime substitution list that corrects words or phrases the ASR model consistently gets wrong for a specific user. It runs as a post-processing stage after ITN, before grammar correction.

## Why it exists

ASR models make systematic errors that are user- or domain-specific:

- Homophones the model picks the wrong form of ("maine" when you mean "main")
- Proper nouns the model has never seen ("Szymanski" → "Shimanski")
- Brand or product names outside the training distribution ("Verba" → "Verbal")
- Domain jargon that collides with common words ("linting" → "listing" for a user whose model is biased toward British English)

nlprule handles universal English grammar errors. User vocab handles personal ones.

## Storage

Persisted to `{data_dir}/vocab.json` where `data_dir` is the platform app data directory:

- **Desktop**: `~/.local/share/verba/` (Linux), `~/Library/Application Support/com.alexb151.verba/` (macOS)
- **Android**: internal app storage, same directory used for history and config

File format — a JSON array of entry objects:

```json
[
  { "from": "maine", "to": "main" },
  { "from": "git hub", "to": "GitHub" },
  { "from": "shimanski", "to": "Szymanski" }
]
```

## Matching rules

- **Case-insensitive input match**: "maine", "Maine", and "MAINE" all trigger the same entry.
- **Word-boundary match**: "maine" matches the word "maine" but not "mainelander". Implemented with `\b` word boundaries in the substitution pass.
- **Exact output**: the `to` field is inserted verbatim. If you want "GitHub" you write "GitHub".
- **Multi-word phrases**: "git hub" matches the two-word sequence as a unit.
- **Order**: entries are applied in the order they are stored. If two entries could overlap, earlier entries win.

## Pipeline position

```
Raw ASR output
  → Filler removal
  → ITN (numbers, dates, ordinals)
  → User vocab substitution      ← here
  → Grammar correction (nlprule)
  → Final cleanup
```

Running after ITN ensures numbers have already been normalised before substitution. Running before grammar lets the grammar checker see correctly spelled words.

## Tauri commands

| Command | Arguments | Returns |
|---|---|---|
| `get_vocab_entries` | — | `Vec<VocabEntry>` |
| `add_vocab_entry` | `from: String, to: String` | `Result<(), String>` |
| `remove_vocab_entry` | `from: String` | `Result<(), String>` |

`VocabEntry` is `{ from: String, to: String }`.

`add_vocab_entry` validates that `from` is non-empty and normalises it to lowercase before storing. Duplicate `from` values replace the existing entry.

`remove_vocab_entry` matches on the stored (lowercased) `from` value.

## UI

A section in the Settings tab, below Voice Enrollment. Shows the current list as rows of `from → to` pairs with a delete button per row. An inline add form accepts the two fields and calls `add_vocab_entry` on submit.

Changes take effect immediately for the next transcription — no restart needed. The substitution stage reads from a `OnceLock<RwLock<Vec<VocabEntry>>>` that is reloaded after each write command.

## Limitations

- **No context awareness**: substitutions fire on every match regardless of surrounding words. "maine" → "main" will also convert a genuine reference to Maine. For context-sensitive corrections, the right tool is a custom nlprule rule in the fork.
- **No regex**: patterns are literal strings with word-boundary matching only. Complex patterns belong in nlprule.
- **Ordering matters for overlapping phrases**: if you have both "git hub actions" and "git hub", put the longer phrase first.
