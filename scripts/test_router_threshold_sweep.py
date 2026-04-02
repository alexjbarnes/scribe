#!/usr/bin/env python3
"""
Threshold sweep for electra-small-discriminator-CoLA.

Sweeps P(acceptable) thresholds from 0.60 to 0.95 in steps of 0.05,
showing the precision/recall tradeoff curve.

Useful for deciding where to set the threshold in production — the goal is
zero FP (no clean technical sentences sent to corrector) while maximizing TP.

Usage:
    python3 scripts/test_router_threshold_sweep.py
    python3 scripts/test_router_threshold_sweep.py --model pszemraj/deberta-v3-xsmall-CoLA
"""

import argparse
import time

from transformers import pipeline

TEST_CASES = [
    ("your going to love this new feature",                   "fix"),
    ("their is a problem with the build",                     "fix"),
    ("we need to look at there approach",                     "fix"),
    ("its going to take longer than expected",                "fix"),
    ("I want to go to store before the meeting",              "fix"),
    ("we need a update to the documentation",                 "fix"),
    ("yesterday I remove all the old log files",              "fix"),
    ("last week we push the release and it broke",            "fix"),
    ("earlier today I fix the bug and deploy it",             "fix"),
    ("he walk to the office every day",                       "fix"),
    ("we was looking at the data yesterday",                  "fix"),
    ("I should of called them earlier",                       "fix"),
    ("could of been a lot worse",                             "fix"),
    ("so I was thinking we could look at the the data again", "fix"),
    ("I I just wanted to say the deployment went well",       "fix"),
    ("The quick brown fox jumps over the lazy dog.",          "preserve"),
    ("I want to schedule a meeting for tomorrow.",            "preserve"),
    ("We deployed the new API endpoint yesterday.",           "preserve"),
    ("Please send me the report by end of day.",              "preserve"),
    ("The function returns a list of strings.",               "preserve"),
    ("we remove old logs every Friday",                       "preserve"),
    ("The PR is in review and CI is green.",                  "preserve"),
    ("API endpoints need to be REST-compliant.",              "preserve"),
]


def p_acceptable(result: dict) -> float:
    label = result["label"].lower()
    score = result["score"]
    return 1.0 - score if "unacceptable" in label else score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="pszemraj/electra-small-discriminator-CoLA",
    )
    args = parser.parse_args()

    print(f"\nLoading {args.model} ...")
    clf = pipeline("text-classification", model=args.model)
    print("Loaded.\n")

    # Score all sentences once
    scores = []
    for text, category in TEST_CASES:
        t0 = time.perf_counter()
        raw = clf(text)[0]
        ms = (time.perf_counter() - t0) * 1000
        pa = p_acceptable(raw)
        scores.append((text, category, pa, ms))

    # Print per-sentence scores (sorted by p_acceptable)
    print(f"{'P(acc)':>6}  {'Cat':8}  Text")
    print("─" * 72)
    for text, cat, pa, ms in sorted(scores, key=lambda x: x[2]):
        inp = text[:54] + "…" if len(text) > 54 else text
        print(f"  {pa:.3f}  {cat:<8}  {inp}  ({ms:.0f}ms)")

    # Sweep thresholds
    thresholds = [round(t, 2) for t in [x * 0.05 for x in range(12, 20)]]  # 0.60 to 0.95

    print(f"\n{'Threshold':>10}  TP  TN  FP  FN   Acc   Prec  Recall   F1  Routes")
    print("─" * 70)
    for threshold in thresholds:
        tp = tn = fp = fn = 0
        routed = 0
        for text, category, pa, _ in scores:
            route = pa < threshold
            if route:
                routed += 1
            if   category == "fix"      and route:  tp += 1
            elif category == "preserve" and not route: tn += 1
            elif category == "preserve" and route:  fp += 1
            else:                                   fn += 1

        n = len(scores)
        acc  = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0

        marker = " <--" if fp == 0 else ""
        print(f"  {threshold:.2f}       {tp:2}  {tn:2}  {fp:2}  {fn:2}"
              f"   {acc:.0%}   {prec:.0%}   {rec:.0%}   {f1:.2f}"
              f"  ({routed}/{n}){marker}")


if __name__ == "__main__":
    main()
