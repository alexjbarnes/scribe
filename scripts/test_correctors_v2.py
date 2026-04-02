#!/usr/bin/env python3
"""
Extended corrector comparison.

Tests multiple grammar/fluency correction models against the same ASR
error test cases.

Models:
  - visheratin/t5-efficient-tiny-grammar-correction  (15.6M, MIT)  [baseline]
  - visheratin/t5-efficient-mini-grammar-correction  (31.2M, MIT)
  - Unbabel/gec-t5_small                             (60M,   Apache 2.0, F0.5=60.7 CoNLL-14)

Unbabel requires a "gec: " prefix on inputs.

Usage:
    python3 scripts/test_correctors_v2.py
    python3 scripts/test_correctors_v2.py --models tiny mini unbabel
    python3 scripts/test_correctors_v2.py --runs 5
"""

import argparse
import statistics
import time

from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline

# (input, expected, category)
TEST_CASES = [
    # Homophones
    ("your going to love this new feature",       "you're going to love this new feature",        "fix/homophone"),
    ("their is a problem with the build",         "there is a problem with the build",            "fix/homophone"),
    ("we need to look at there approach",         "we need to look at their approach",            "fix/homophone"),
    ("its going to take longer than expected",    "it's going to take longer than expected",      "fix/homophone"),
    # Articles
    ("I want to go to store before the meeting",  "I want to go to the store before the meeting", "fix/article"),
    ("we need a update to the documentation",     "we need an update to the documentation",       "fix/article"),
    # Verb tense
    ("yesterday I remove all the old log files",  "yesterday I removed all the old log files",    "fix/tense"),
    ("last week we push the release and it broke","last week we pushed the release and it broke", "fix/tense"),
    ("earlier today I fix the bug and deploy it", "earlier today I fixed the bug and deployed it","fix/tense"),
    # Agreement
    ("he walk to the office every day",           "he walks to the office every day",             "fix/agreement"),
    ("we was looking at the data yesterday",      "we were looking at the data yesterday",        "fix/agreement"),
    # Informal
    ("I should of called them earlier",           "I should have called them earlier",            "fix/informal"),
    ("could of been a lot worse",                 "could have been a lot worse",                  "fix/informal"),
    # ASR duplicates
    ("so I was thinking we could look at the the data again",
     "so I was thinking we could look at the data again",                                         "fix/asr-dup"),
    ("I I just wanted to say the deployment went well",
     "I just wanted to say the deployment went well",                                             "fix/asr-dup"),
    # Preserve — clean text
    ("The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog.", "preserve/clean"),
    ("I want to schedule a meeting for tomorrow.",   "I want to schedule a meeting for tomorrow.",   "preserve/clean"),
    ("We deployed the new API endpoint yesterday.",  "We deployed the new API endpoint yesterday.",  "preserve/tech"),
    ("Please send me the report by end of day.",     "Please send me the report by end of day.",     "preserve/clean"),
    ("The function returns a list of strings.",      "The function returns a list of strings.",      "preserve/tech"),
    ("we remove old logs every Friday",              "we remove old logs every Friday",              "preserve/habitual"),
    ("The PR is in review and CI is green.",         "The PR is in review and CI is green.",         "preserve/tech"),
    ("API endpoints need to be REST-compliant.",     "API endpoints need to be REST-compliant.",     "preserve/tech"),
]

MODELS = {
    "tiny":   ("visheratin/t5-efficient-tiny-grammar-correction",  None),
    "mini":   ("visheratin/t5-efficient-mini-grammar-correction",  None),
    "unbabel": ("Unbabel/gec-t5_small",                            "gec: "),
}


def load_corrector(model_id: str):
    print(f"  Loading {model_id} ...")
    t0 = time.perf_counter()
    model     = T5ForConditionalGeneration.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"  loaded in {(time.perf_counter() - t0)*1000:.0f}ms")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def run_corrector(corrector, text: str, prefix: str | None, runs: int) -> tuple[str, float]:
    inp = (prefix + text) if prefix else text
    times = []
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = corrector(inp, max_length=128)[0]["generated_text"].strip()
        times.append((time.perf_counter() - t0) * 1000)
        out = result
    return out, statistics.median(times)


def evaluate(model_id: str, corrector, prefix: str | None, runs: int) -> dict:
    results = []
    for text, expected, category in TEST_CASES:
        output, ms = run_corrector(corrector, text, prefix, runs)
        changed = output.lower() != text.lower()
        correct = output.lower() == expected.lower()
        results.append({
            "text": text, "expected": expected, "output": output,
            "category": category, "changed": changed, "correct": correct, "ms": ms,
        })

    fix_cases      = [r for r in results if r["category"].startswith("fix")]
    preserve_cases = [r for r in results if r["category"].startswith("preserve")]

    # Correct fix = changed the text (we don't enforce exact match, just changed ≠ unchanged)
    # Over-correction = changed a preserve case
    fixed_correctly  = sum(1 for r in fix_cases      if r["changed"])
    over_corrections = sum(1 for r in preserve_cases if r["changed"])
    missed_errors    = sum(1 for r in fix_cases      if not r["changed"])

    all_ms = [r["ms"] for r in results]

    short = model_id.split("/")[-1]
    col   = 46

    print(f"\n{'═'*72}")
    print(f"  {short}  (prefix={repr(prefix)})")
    print(f"{'═'*72}")

    for group_label, group in [("FIX cases", fix_cases), ("PRESERVE cases", preserve_cases)]:
        print(f"\n  ── {group_label}")
        for r in group:
            marker = "~" if r["changed"] else "="
            inp = r["text"][:col] + "…" if len(r["text"]) > col else r["text"]
            out = r["output"][:col] + "…" if len(r["output"]) > col else r["output"]
            print(f"  {marker} {inp:<{col}}  →  {out:<{col}}  {r['ms']:.0f}ms  {r['category']}")

    print()
    print(f"  Fixed (changed):     {fixed_correctly}/{len(fix_cases)}")
    print(f"  Missed errors:       {missed_errors}/{len(fix_cases)}")
    print(f"  Over-corrected:      {over_corrections}/{len(preserve_cases)} preserve cases")
    print(f"  Latency: median={statistics.median(all_ms):.0f}ms"
          f"  p95={sorted(all_ms)[int(len(all_ms)*0.95)]:.0f}ms"
          f"  max={max(all_ms):.0f}ms")

    return {
        "model": short,
        "fixed": fixed_correctly,
        "fix_total": len(fix_cases),
        "missed": missed_errors,
        "over": over_corrections,
        "preserve_total": len(preserve_cases),
        "median_ms": statistics.median(all_ms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
    )
    parser.add_argument("--runs", type=int, default=3,
                        help="Timed runs per input (default: 3)")
    args = parser.parse_args()

    print(f"\nLoading {len(args.models)} model(s) ...\n")
    loaded = []
    for key in args.models:
        model_id, prefix = MODELS[key]
        try:
            c = load_corrector(model_id)
            loaded.append((model_id, c, prefix))
        except Exception as e:
            print(f"  FAILED to load {model_id}: {e}")

    summaries = []
    for model_id, c, prefix in loaded:
        s = evaluate(model_id, c, prefix, args.runs)
        summaries.append(s)

    print(f"\n{'─'*72}")
    print("Summary")
    print(f"{'─'*72}")
    print(f"  {'Model':<46}  Fixed  Missed  OverCorr  ms")
    for s in summaries:
        print(f"  {s['model']:<46}"
              f"  {s['fixed']}/{s['fix_total']}"
              f"     {s['missed']}/{s['fix_total']}"
              f"     {s['over']}/{s['preserve_total']}"
              f"       {s['median_ms']:.0f}ms")


if __name__ == "__main__":
    main()
