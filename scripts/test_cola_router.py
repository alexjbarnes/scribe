#!/usr/bin/env python3
"""
Test CoLA classifiers as the routing stage instead of DistilGPT2.

Hypothesis: a linguistic acceptability classifier should handle technical
vocabulary better than a perplexity scorer, since it learns grammatical
structure rather than word frequency.

Tests two CoLA models as routers:
  - pszemraj/xtremedistil-l12-h384-uncased-CoLA  (33M, MIT, MCC=0.540)
  - pszemraj/deberta-v3-xsmall-CoLA              (70.8M, MIT, MCC=0.590)

Each is compared on the same test cases. The corrector (t5-efficient-tiny)
only runs on sentences classified as unacceptable.

Usage:
    python3 scripts/test_cola_router.py
    python3 scripts/test_cola_router.py --threshold 0.7   # require 70% confidence
"""

import argparse
import statistics
import time

from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline

TEST_CASES = [
    # (input, category)
    # fix      = contains an error, should be routed to corrector
    # preserve = clean text, should pass through unchanged
    ("your going to love this new feature",           "fix"),
    ("their is a problem with the build",             "fix"),
    ("we need to look at there approach",             "fix"),
    ("its going to take longer than expected",        "fix"),
    ("I want to go to store before the meeting",      "fix"),
    ("we need a update to the documentation",         "fix"),
    ("yesterday I remove all the old log files",      "fix"),
    ("last week we push the release and it broke",    "fix"),
    ("earlier today I fix the bug and deploy it",     "fix"),
    ("he walk to the office every day",               "fix"),
    ("we was looking at the data yesterday",          "fix"),
    ("I should of called them earlier",               "fix"),
    ("could of been a lot worse",                     "fix"),
    ("so I was thinking we could look at the the data again", "fix"),
    ("I I just wanted to say the deployment went well",       "fix"),
    # preserve
    ("The quick brown fox jumps over the lazy dog.",  "preserve"),
    ("I want to schedule a meeting for tomorrow.",    "preserve"),
    ("We deployed the new API endpoint yesterday.",   "preserve"),
    ("Please send me the report by end of day.",      "preserve"),
    ("The function returns a list of strings.",       "preserve"),
    ("we remove old logs every Friday",               "preserve"),
    ("The PR is in review and CI is green.",          "preserve"),
    ("API endpoints need to be REST-compliant.",      "preserve"),
]

COLA_MODELS = [
    "pszemraj/xtremedistil-l12-h384-uncased-CoLA",
    "pszemraj/deberta-v3-xsmall-CoLA",
]


def load_corrector():
    repo = "visheratin/t5-efficient-tiny-grammar-correction"
    print(f"Loading corrector ({repo}) ...")
    t0 = time.perf_counter()
    model = T5ForConditionalGeneration.from_pretrained(repo)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    print(f"  loaded in {(time.perf_counter() - t0)*1000:.0f}ms")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def load_cola(model_id: str):
    print(f"Loading CoLA classifier ({model_id}) ...")
    t0 = time.perf_counter()
    clf = pipeline("text-classification", model=model_id)
    print(f"  loaded in {(time.perf_counter() - t0)*1000:.0f}ms")
    return clf


def run_cola_pipeline(
    text: str,
    cola_clf,
    corrector,
    threshold: float,
) -> dict:
    t_start = time.perf_counter()

    # Stage 1: CoLA classification
    t1 = time.perf_counter()
    result = cola_clf(text)[0]
    score_ms = (time.perf_counter() - t1) * 1000

    label = result["label"]        # "acceptable" or "unacceptable"
    confidence = result["score"]   # confidence in that label

    # Route to corrector if:
    #   - classified as "unacceptable" with any confidence, OR
    #   - classified as "acceptable" but with low confidence (< threshold)
    if label == "unacceptable":
        p_acceptable = 1.0 - confidence
    else:
        p_acceptable = confidence

    should_correct = p_acceptable < threshold

    # Stage 2: correction (conditional)
    correct_ms = 0.0
    if should_correct:
        t2 = time.perf_counter()
        output = corrector(text, max_length=128)[0]["generated_text"].strip()
        correct_ms = (time.perf_counter() - t2) * 1000
    else:
        output = text

    total_ms = (time.perf_counter() - t_start) * 1000

    return {
        "output": output,
        "routed": should_correct,
        "label": label,
        "p_acceptable": p_acceptable,
        "score_ms": score_ms,
        "correct_ms": correct_ms,
        "total_ms": total_ms,
    }


def evaluate(model_id: str, cola_clf, corrector, threshold: float):
    results = []
    tp = tn = fp = fn = 0

    for text, category in TEST_CASES:
        r = run_cola_pipeline(text, cola_clf, corrector, threshold)

        if category == "fix" and r["routed"]:
            verdict = "TP"
            tp += 1
        elif category == "preserve" and not r["routed"]:
            verdict = "TN"
            tn += 1
        elif category == "preserve" and r["routed"]:
            verdict = "FP"
            fp += 1
        else:
            verdict = "FN"
            fn += 1

        results.append({**r, "input": text, "category": category, "verdict": verdict})

    n = len(results)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    routed  = [r for r in results if r["routed"]]
    passed  = [r for r in results if not r["routed"]]
    all_t   = [r["total_ms"] for r in results]
    pass_t  = [r["total_ms"] for r in passed]
    route_t = [r["total_ms"] for r in routed]

    short_id = model_id.split("/")[-1]
    print(f"\n{'═' * 72}")
    print(f"  {short_id}  |  threshold={threshold}")
    print(f"{'═' * 72}")

    for r in results:
        routed_tag = "ROUTE" if r["routed"] else "PASS "
        inp = r["input"][:52] + "…" if len(r["input"]) > 52 else r["input"]
        changed = r["output"].lower() != r["input"].lower()
        out_tag = "~" if changed else "="
        p = r["p_acceptable"]

        if r["routed"]:
            out_str = r["output"][:44] + "…" if len(r["output"]) > 44 else r["output"]
            print(f"  [{r['verdict']}] {routed_tag} p={p:.2f}  {inp}")
            print(f"               {out_tag} {out_str}  ({r['total_ms']:.0f}ms)")
        else:
            print(f"  [{r['verdict']}] {routed_tag} p={p:.2f}  {inp}  ({r['total_ms']:.0f}ms)")

    print()
    print(f"  Routing: TP={tp} TN={tn} FP={fp} FN={fn}"
          f"  acc={accuracy:.0%} prec={precision:.0%} recall={recall:.0%} F1={f1:.2f}")
    print(f"  Latency all={statistics.median(all_t):.0f}ms median"
          + (f"  passed={statistics.median(pass_t):.0f}ms" if pass_t else "")
          + (f"  routed={statistics.median(route_t):.0f}ms" if route_t else ""))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy,
            "f1": f1, "model": short_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="P(acceptable) below this → route to corrector (default: 0.85)")
    args = parser.parse_args()

    print()
    corrector = load_corrector()
    classifiers = [(m, load_cola(m)) for m in COLA_MODELS]
    print()

    summaries = []
    for model_id, clf in classifiers:
        s = evaluate(model_id, clf, corrector, args.threshold)
        summaries.append(s)

    print(f"\n{'─' * 72}")
    print("Summary")
    print(f"{'─' * 72}")
    print(f"  {'Model':<45}  TP  TN  FP  FN  Acc   F1")
    for s in summaries:
        print(f"  {s['model']:<45}  {s['tp']:2}  {s['tn']:2}  {s['fp']:2}  {s['fn']:2}"
              f"  {s['accuracy']:.0%}  {s['f1']:.2f}")


if __name__ == "__main__":
    main()
