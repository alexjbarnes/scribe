#!/usr/bin/env python3
"""
Extended router comparison: test additional CoLA classifiers as routing stage.

New candidates vs the previously tested pair:
  - pszemraj/electra-small-discriminator-CoLA   (13.5M, Apache 2.0, MCC=0.551)
  - pszemraj/electra-base-discriminator-CoLA    (110M,  Apache 2.0, MCC=0.658, ONNX)
  - mrm8488/deberta-v3-small-finetuned-cola     (143M,  MIT,        MCC=0.633)

Previously tested (included for comparison):
  - pszemraj/xtremedistil-l12-h384-uncased-CoLA (33M,  MIT, MCC=0.540)
  - pszemraj/deberta-v3-xsmall-CoLA             (70.8M, MIT, MCC=0.590)

Usage:
    python3 scripts/test_routers_v2.py
    python3 scripts/test_routers_v2.py --threshold 0.8
    python3 scripts/test_routers_v2.py --models electra-small electra-base
"""

import argparse
import statistics
import time

from transformers import pipeline

TEST_CASES = [
    # (input, category)
    # fix      = contains a grammar error
    # preserve = clean text, should pass through
    ("your going to love this new feature",                 "fix"),
    ("their is a problem with the build",                   "fix"),
    ("we need to look at there approach",                   "fix"),
    ("its going to take longer than expected",              "fix"),
    ("I want to go to store before the meeting",            "fix"),
    ("we need a update to the documentation",               "fix"),
    ("yesterday I remove all the old log files",            "fix"),
    ("last week we push the release and it broke",          "fix"),
    ("earlier today I fix the bug and deploy it",           "fix"),
    ("he walk to the office every day",                     "fix"),
    ("we was looking at the data yesterday",                "fix"),
    ("I should of called them earlier",                     "fix"),
    ("could of been a lot worse",                           "fix"),
    ("so I was thinking we could look at the the data again", "fix"),
    ("I I just wanted to say the deployment went well",     "fix"),
    # preserve
    ("The quick brown fox jumps over the lazy dog.",        "preserve"),
    ("I want to schedule a meeting for tomorrow.",          "preserve"),
    ("We deployed the new API endpoint yesterday.",         "preserve"),
    ("Please send me the report by end of day.",            "preserve"),
    ("The function returns a list of strings.",             "preserve"),
    ("we remove old logs every Friday",                     "preserve"),
    ("The PR is in review and CI is green.",                "preserve"),
    ("API endpoints need to be REST-compliant.",            "preserve"),
]

MODELS = {
    "electra-small": "pszemraj/electra-small-discriminator-CoLA",
    "electra-base":  "pszemraj/electra-base-discriminator-CoLA",
    "deberta-small": "mrm8488/deberta-v3-small-finetuned-cola",
    "xtremedistil":  "pszemraj/xtremedistil-l12-h384-uncased-CoLA",
    "deberta-xsmall": "pszemraj/deberta-v3-xsmall-CoLA",
}


def load_router(model_id: str):
    print(f"  Loading {model_id} ...")
    t0 = time.perf_counter()
    clf = pipeline("text-classification", model=model_id)
    print(f"  loaded in {(time.perf_counter() - t0)*1000:.0f}ms")
    return clf


def p_acceptable(result: dict) -> float:
    """Return P(acceptable) regardless of which label the model predicted."""
    label = result["label"].lower()
    score = result["score"]
    if "unacceptable" in label:
        return 1.0 - score
    return score


def evaluate(model_id: str, clf, threshold: float) -> dict:
    tp = tn = fp = fn = 0
    score_times = []
    routed_times = []
    pass_times = []

    rows = []
    for text, category in TEST_CASES:
        t0 = time.perf_counter()
        raw = clf(text)[0]
        elapsed = (time.perf_counter() - t0) * 1000
        score_times.append(elapsed)

        pa = p_acceptable(raw)
        routed = pa < threshold

        if category == "fix" and routed:
            verdict = "TP"; tp += 1
        elif category == "preserve" and not routed:
            verdict = "TN"; tn += 1
        elif category == "preserve" and routed:
            verdict = "FP"; fp += 1
        else:
            verdict = "FN"; fn += 1

        rows.append({
            "text": text, "category": category, "verdict": verdict,
            "routed": routed, "p_acceptable": pa, "ms": elapsed,
        })
        if routed:
            routed_times.append(elapsed)
        else:
            pass_times.append(elapsed)

    n = len(rows)
    accuracy  = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0

    short = model_id.split("/")[-1]
    print(f"\n{'═'*70}")
    print(f"  {short}  |  threshold={threshold}")
    print(f"{'═'*70}")
    for r in rows:
        tag  = "ROUTE" if r["routed"] else "PASS "
        inp  = r["text"][:54] + "…" if len(r["text"]) > 54 else r["text"]
        print(f"  [{r['verdict']}] {tag} p={r['p_acceptable']:.3f}  {inp}  ({r['ms']:.0f}ms)")
    print()
    print(f"  Routing: TP={tp} TN={tn} FP={fp} FN={fn}"
          f"  acc={accuracy:.0%} prec={precision:.0%} recall={recall:.0%} F1={f1:.2f}")
    print(f"  Scorer latency: median={statistics.median(score_times):.0f}ms"
          + (f"  passed={statistics.median(pass_times):.0f}ms" if pass_times else "")
          + (f"  routed={statistics.median(routed_times):.0f}ms" if routed_times else ""))

    return {
        "model": short, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "f1": f1,
        "median_ms": statistics.median(score_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which models to test (default: all)",
    )
    args = parser.parse_args()

    print(f"\nLoading {len(args.models)} model(s) ...\n")
    loaded = []
    for key in args.models:
        model_id = MODELS[key]
        try:
            clf = load_router(model_id)
            loaded.append((model_id, clf))
        except Exception as e:
            print(f"  FAILED to load {model_id}: {e}")

    summaries = []
    for model_id, clf in loaded:
        s = evaluate(model_id, clf, args.threshold)
        summaries.append(s)

    print(f"\n{'─'*70}")
    print(f"Summary  (threshold={args.threshold})")
    print(f"{'─'*70}")
    print(f"  {'Model':<48}  TP  TN  FP  FN   Acc   F1  ms")
    for s in summaries:
        print(f"  {s['model']:<48}  {s['tp']:2}  {s['tn']:2}  {s['fp']:2}  {s['fn']:2}"
              f"  {s['accuracy']:.0%}  {s['f1']:.2f}  {s['median_ms']:.0f}ms")


if __name__ == "__main__":
    main()
