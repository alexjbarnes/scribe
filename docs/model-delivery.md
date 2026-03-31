# Model and Rule Data Delivery

## Overview

Models and rule data (nlprule binaries, spell correction dictionaries) are hosted externally and downloaded on first launch. Downloads are gated behind user authentication and an active subscription.

## Architecture

```
App -> Supabase Auth (login, get JWT)
App -> Supabase Edge Function (JWT auto-validated, check subscription) -> R2 presigned URL
App -> Cloudflare R2 (download file directly, free egress)
```

All auth and billing state lives in Supabase. Cloudflare R2 is a dumb file store with no public access. The Edge Function is the only thing that can generate download URLs.

## Components

### Supabase Auth

Handles user signup, login, and JWT issuance. The app stores the JWT locally and refreshes it on expiry. Supabase Edge Functions validate the JWT automatically from the `Authorization` header, so the Edge Function itself contains no auth logic.

### Supabase DB

Users table extended with:
- `stripe_customer_id` -- set when the user subscribes
- `subscription_status` -- `active`, `trialing`, `past_due`, `cancelled`
- `subscription_tier` -- if we add tiers later

Subscription status is updated by Stripe webhooks (see below). It can also be embedded as a custom claim in the JWT via a Supabase database hook, so the Edge Function can check it without a DB query.

### Stripe

Handles billing. User subscribes via Stripe Checkout (web link opened from the app). Stripe sends webhooks to a Supabase Edge Function that updates `subscription_status` in the DB.

Key webhook events to handle:
- `checkout.session.completed` -- link `stripe_customer_id` to user, set status `active`
- `customer.subscription.updated` -- status changes (active, past_due, cancelled)
- `customer.subscription.deleted` -- set status `cancelled`
- `invoice.payment_failed` -- set status `past_due`

### Supabase Edge Function: generate-download-url

Called by the app to get a presigned R2 URL for a model or rule file.

```
POST /functions/v1/generate-download-url
Authorization: Bearer <supabase_jwt>
Body: { "file": "models/parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx" }
```

The function:
1. JWT is validated automatically by Supabase (no code needed)
2. Read the user's subscription status from the JWT custom claim or DB
3. If not active, return 403
4. Generate a presigned R2 GET URL (expires in 15 minutes)
5. Return `{ "url": "https://r2-bucket.../encoder.int8.onnx?X-Amz-..." }`

R2 presigned URLs use the S3-compatible API. The Edge Function needs `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` as environment variables.

### Cloudflare R2

Private bucket, no public access. Stores:
- Model files (ONNX weights, tokenizer data, tokens)
- nlprule binary data (en_rules.bin, en_tokenizer.bin)
- Spell correction dictionaries

R2 egress is free. This is the main reason to use R2 over Supabase Storage for large files like the 670MB Parakeet model.

## App-side flow

### First launch (no account)

1. App opens to a login/signup screen
2. User creates account via Supabase Auth (email + password or magic link)
3. User subscribes via Stripe Checkout (web link)
4. App stores JWT, proceeds to model download

### Model download

1. App requests presigned URLs for each model file from the Edge Function
2. Downloads files directly from R2 with progress reporting
3. Saves to app data directory
4. On completion, loads the model and starts the engine

### Subsequent launches

1. App checks for locally cached model files
2. If present and valid, skips download, starts engine immediately
3. JWT refresh happens in the background
4. If subscription has lapsed, the app still works with already-downloaded models (no phone-home DRM on cached files)

### Model updates

When we ship a new model version:
1. App checks a manifest endpoint for the latest version
2. If newer than local, prompts user to download
3. Old model files can be cleaned up after successful download

## File layout in R2

```
models/
  parakeet-tdt-0.6b-v3-int8/
    encoder.int8.onnx    (652 MB)
    decoder.int8.onnx    (12 MB)
    joiner.int8.onnx     (6.4 MB)
    tokens.txt           (94 KB)
rules/
  nlprule/
    en_rules.bin         (7.2 MB)
    en_tokenizer.bin     (11.1 MB)
  spelling/
    en_dictionary.txt
manifest.json            (current versions, file checksums)
```

## Manifest

A small JSON file the app checks to know what to download and whether updates are available.

```json
{
  "version": 1,
  "model": {
    "id": "parakeet-tdt-0.6b-v3-int8",
    "version": "2026.03",
    "files": [
      { "path": "encoder.int8.onnx", "sha256": "abc123...", "bytes": 652000000 },
      { "path": "decoder.int8.onnx", "sha256": "def456...", "bytes": 12000000 },
      { "path": "joiner.int8.onnx", "sha256": "789abc...", "bytes": 6400000 },
      { "path": "tokens.txt", "sha256": "012def...", "bytes": 94000 }
    ]
  },
  "rules": {
    "nlprule_version": "2026.03",
    "spelling_version": "2026.03",
    "files": [
      { "path": "nlprule/en_rules.bin", "sha256": "...", "bytes": 7200000 },
      { "path": "nlprule/en_tokenizer.bin", "sha256": "...", "bytes": 11100000 },
      { "path": "spelling/en_dictionary.txt", "sha256": "...", "bytes": 500000 }
    ]
  }
}
```

The manifest itself can be fetched without auth (it contains no sensitive data, just file names and checksums). This lets the app check for updates without requiring an active session.

## Decisions

**No phone-home DRM.** Once models are downloaded, they work offline forever. If the subscription lapses, the user keeps what they have but cannot download updates or new rule data. This avoids bricking the app for users who lose connectivity or forget to renew.

**Single model, no selection UI.** The app ships one model (Parakeet V3 INT8). No model picker, no user-facing model management. The manifest tells the app what to download. If we ship a better model later, it appears as an update.

**Stripe over in-app purchase.** ~97% revenue retention vs ~70% through app stores. Subscription management happens on the web via Stripe Customer Portal. Risk: Apple/Google may require in-app purchase for digital content consumed in the app. Mitigated by framing the subscription as account access, not a content purchase.

**R2 over Supabase Storage.** Free egress on R2. A 670MB model download per user on Supabase Storage would cost significantly more at scale.
