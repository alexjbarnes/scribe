# Remote Data Files

Documents the plan to move large bundled data files out of the APK and onto
R2, and records the decisions made along the way.

## Current state

Five files are currently embedded via `include_bytes!()` and baked into the
compiled binary:

| File | Size | Used for |
|------|------|----------|
| `silero_vad.onnx` | 644 KB | Voice activity detection (core) |
| `speaker_embed.onnx` | 29.6 MB | Speaker verification (optional feature) |
| `en_rules.bin` | 22.6 MB | Grammar correction (post-processing) |
| `en_tokenizer.bin` | 11.6 MB | Grammar tokenization (post-processing) |
| `frequency_dictionary_en_82_765.txt` | 1.3 MB | Spell correction (post-processing) |

**Total baked into APK: ~65 MB**

## Decision: keep silero_vad.onnx bundled

`silero_vad.onnx` stays embedded. Reasons:

- 644 KB against a 106 MB APK is negligible.
- Silero VAD v4 is stable — no active development pushing new versions.
  Updates will be infrequent and warrant a binary release anyway.
- It is a hard boot dependency. Recording cannot start without it. Requiring
  a network download before first use adds a failure mode and friction for no
  real benefit.
- It is universal — every user gets the same file, it is not user-selectable
  like the ASR models.

Everything else moves to R2.

## Files to move to R2

| File | Size | Download timing |
|------|------|----------------|
| `speaker_embed.onnx` | 29.6 MB | Lazy — only when user enrolls a speaker |
| `en_rules.bin` | 22.6 MB | Background — after engine starts |
| `en_tokenizer.bin` | 11.6 MB | Background — after engine starts |
| `frequency_dictionary_en_82_765.txt` | 1.3 MB | Background — after engine starts |

**APK size after migration: ~7 MB (excl. ASR models, which are already remote)**

## Architecture

### Manifest

A single `manifest.json` on R2 describes all downloadable assets — both
the data files above and the ASR models. The app fetches this on launch
and uses it as the source of truth for download URLs.

```json
{
  "version": 1,
  "data_files": [
    { "id": "speaker_embed",  "filename": "speaker_embed.onnx",                     "url": "https://...", "bytes": 31040512 },
    { "id": "en_rules",       "filename": "en_rules.bin",                           "url": "https://...", "bytes": 23707648 },
    { "id": "en_tokenizer",   "filename": "en_tokenizer.bin",                       "url": "https://...", "bytes": 12189696 },
    { "id": "freq_dict",      "filename": "frequency_dictionary_en_82_765.txt",     "url": "https://...", "bytes": 1392640  }
  ],
  "models": {
    "parakeet-tdt-0.6b-v3-int8": [
      { "role": "encoder", "url": "https://...", "bytes": 12345678 },
      ...
    ]
  }
}
```

### Download timing

```
App launch
  └─ Fetch manifest
  └─ Engine init (silero_vad.onnx already on disk — bundled)
  └─ Engine ready
       └─ Background: download grammar + spelling files if absent
            └─ Post-processing pipeline picks them up on next use

User enrolls speaker
  └─ Trigger: download speaker_embed.onnx if absent
  └─ Proceed with enrollment
```

### Graceful degradation

Each post-processing stage checks for its data file at startup. If absent,
that stage is skipped silently — transcription still works, just without
grammar polish or spell correction. The UI does not need to surface this.

Speaker verification is already optional by design — enrollment is a
deliberate user action.

### Auth upgrade path

Phase 1 (now, public bucket):
- Manifest URL is a public R2 URL, no auth header needed.
- Download URLs in the manifest are also public R2 URLs.

Phase 2 (when auth is added):
- Manifest URL changes to a service endpoint (e.g. `https://api.verba.app/v1/manifest`).
- The fetch adds `Authorization: Bearer <jwt>`.
- Service verifies the JWT and returns the same manifest structure, but
  with presigned R2 URLs in place of the public ones.
- The download function (`client.get(&url).send()`) does not change.
  Only the manifest fetch gains the auth header.

A single function signature handles both phases:

```rust
async fn fetch_manifest(url: &str, token: Option<&str>) -> Result<Manifest>
```

`token` is `None` in phase 1, populated from the auth store in phase 2.

## Implementation plan

Blocked on: R2 bucket creation and file upload.

### Step 1 — R2 setup (infrastructure, outside codebase)
- Create R2 bucket.
- Upload the four data files and all ASR model files.
- Set public access (or generate public URL prefix).
- Record the public base URL.

### Step 2 — Manifest
- Write `manifest.json` with all data file and model entries.
- Upload to R2.
- Add `MANIFEST_URL` constant to `models.rs` (or `config.rs`).

### Step 3 — Manifest fetch
- Add `fetch_manifest(url, token)` async fn to `models.rs`.
- On launch: fetch manifest, store URL map in `ModelManager`.
- Update `ModelDef.files[].url` from the manifest for ASR models.
- Add `DataFileEntry` entries from `manifest.data_files` to `ModelManager`.

### Step 4 — Data file downloads
- Add `ensure_data_file(id)` to `ModelManager` — checks disk, downloads if
  absent, emits progress events.
- Call for grammar/spelling files in a `tokio::spawn` after engine is ready.
- Call for `speaker_embed.onnx` at the start of the speaker enrollment flow.

### Step 5 — Load from disk instead of `include_bytes!`
- `grammar.rs`: replace `include_bytes!` with `std::fs::read()` from the
  data dir. If file absent, return `None` and skip the grammar stage.
- `spelling.rs`: same pattern for the frequency dictionary.
- `models.rs`: remove `include_bytes!` for `speaker_embed.onnx`. Load path
  comes from `ModelManager::speaker_embed_path()`.
- Remove the four data files from the repo (or move to a `dev-data/` folder
  for local dev use only).

### Step 6 — Remove from binary
- Delete `include_bytes!` lines.
- Verify APK size reduction.

## Local development

During local dev (no R2), place the data files in the expected on-disk
location manually. The engine will find them and skip the download. A
`just fetch-data` recipe could automate this using the public manifest URL
once step 2 is done.
