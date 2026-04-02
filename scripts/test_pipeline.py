#!/usr/bin/env python3
"""
Two-stage fluency correction pipeline test:

  Stage 1 — DistilGPT2 perplexity scorer (~20-40ms)
    Run a single forward pass to get per-token log-probabilities.
    Flag tokens below a threshold as likely errors.
    If any flagged tokens found → route to stage 2.

  Stage 2 — t5-efficient-tiny correction (~50-150ms)
    Only invoked when stage 1 flags the sentence.
    Produces corrected output.

The routing is the key idea: clean sentences pass through in ~20ms,
only problem sentences pay the full ~70ms correction cost.

Usage:
    python3 scripts/test_pipeline.py
    python3 scripts/test_pipeline.py --threshold -6.0
    python3 scripts/test_pipeline.py --threshold -5.0 --model mini
"""

import argparse
import statistics
import time

import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    pipeline,
)

# ── Test cases ────────────────────────────────────────────────────────────────
# (input, expected_output, category)
# category: "fix" = should be corrected, "preserve" = should pass through
TEST_CASES = [
    # Homophones
    ("your going to love this new feature",           "you're going to love this new feature",            "fix"),
    ("their is a problem with the build",             "there is a problem with the build",                "fix"),
    ("we need to look at there approach",             "we need to look at their approach",                "fix"),
    ("its going to take longer than expected",        "it's going to take longer than expected",          "fix"),
    # Articles
    ("I want to go to store before the meeting",      "I want to go to the store before the meeting",     "fix"),
    ("we need a update to the documentation",         "we need an update to the documentation",          "fix"),
    # Verb tense
    ("yesterday I remove all the old log files",      "yesterday I removed all the old log files",        "fix"),
    ("last week we push the release and it broke",    "last week we pushed the release and it broke",     "fix"),
    ("earlier today I fix the bug and deploy it",     "earlier today I fixed the bug and deployed it",    "fix"),
    # Subject-verb agreement
    ("he walk to the office every day",               "he walks to the office every day",                 "fix"),
    ("we was looking at the data yesterday",          "we were looking at the data yesterday",            "fix"),
    # Spoken constructions
    ("I should of called them earlier",               "I should have called them earlier",                "fix"),
    ("could of been a lot worse",                     "could have been a lot worse",                      "fix"),
    # ASR duplicates
    ("so I was thinking we could look at the the data again", "so I was thinking we could look at the data again", "fix"),
    ("I I just wanted to say the deployment went well",       "I just wanted to say the deployment went well",      "fix"),
    # Clean — should NOT be routed to correction
    ("The quick brown fox jumps over the lazy dog.",  "The quick brown fox jumps over the lazy dog.",     "preserve"),
    ("I want to schedule a meeting for tomorrow.",    "I want to schedule a meeting for tomorrow.",       "preserve"),
    ("We deployed the new API endpoint yesterday.",   "We deployed the new API endpoint yesterday.",      "preserve"),
    ("Please send me the report by end of day.",      "Please send me the report by end of day.",         "preserve"),
    ("The function returns a list of strings.",       "The function returns a list of strings.",          "preserve"),
    ("we remove old logs every Friday",               "we remove old logs every Friday",                  "preserve"),
    ("The PR is in review and CI is green.",          "The PR is in review and CI is green.",             "preserve"),
    ("API endpoints need to be REST-compliant.",      "API endpoints need to be REST-compliant.",         "preserve"),
]


# ── Stage 1: DistilGPT2 perplexity scorer ─────────────────────────────────────

def load_scorer():
    print("Loading DistilGPT2 scorer ...")
    t0 = time.perf_counter()
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()
    print(f"  loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")
    return model, tokenizer


def score_tokens(text: str, model, tokenizer) -> list[tuple[str, float]]:
    """
    Return [(token_str, log_prob), ...] for each token in `text`.
    log_prob is P(token | all preceding tokens) from DistilGPT2.
    Lower = more surprising = more likely wrong.
    """
    enc = tokenizer.encode(text, return_tensors="pt")
    if enc.shape[1] < 2:
        return []

    with torch.no_grad():
        logits = model(enc).logits[0]  # [seq_len, vocab]

    # Position i predicts token i+1, so shift by 1
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    targets = enc[0, 1:]
    token_log_probs = log_probs[torch.arange(len(targets)), targets].tolist()
    token_strings = [tokenizer.decode([t]) for t in targets.tolist()]

    return list(zip(token_strings, token_log_probs))


def route(token_scores: list[tuple[str, float]], threshold: float) -> tuple[bool, list[str]]:
    """
    Return (should_correct, flagged_tokens).
    A sentence is routed to correction if any token is below `threshold`.
    """
    flagged = [tok for tok, lp in token_scores if lp < threshold]
    return bool(flagged), flagged


# ── Stage 2: t5-efficient-tiny corrector ──────────────────────────────────────

def load_corrector(model_size: str):
    repo = f"visheratin/t5-efficient-{model_size}-grammar-correction"
    print(f"Loading {repo} ...")
    t0 = time.perf_counter()
    model = T5ForConditionalGeneration.from_pretrained(repo)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    print(f"  loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(
    text: str,
    scorer_model,
    scorer_tokenizer,
    corrector,
    threshold: float,
) -> dict:
    t_start = time.perf_counter()

    # Stage 1
    t1 = time.perf_counter()
    token_scores = score_tokens(text, scorer_model, scorer_tokenizer)
    score_ms = (time.perf_counter() - t1) * 1000

    should_correct, flagged = route(token_scores, threshold)

    # Stage 2 (conditional)
    correct_ms = 0.0
    if should_correct:
        t2 = time.perf_counter()
        result = corrector(text, max_length=128)[0]["generated_text"]
        correct_ms = (time.perf_counter() - t2) * 1000
    else:
        result = text

    total_ms = (time.perf_counter() - t_start) * 1000

    # Find the min log-prob token for display
    min_token = min(token_scores, key=lambda x: x[1]) if token_scores else ("", 0.0)

    return {
        "output": result,
        "routed": should_correct,
        "flagged": flagged,
        "min_token": min_token,
        "score_ms": score_ms,
        "correct_ms": correct_ms,
        "total_ms": total_ms,
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=-7.0,
                        help="Log-prob threshold for routing (default: -7.0)")
    parser.add_argument("--model", choices=["tiny", "mini"], default="tiny")
    args = parser.parse_args()

    print(f"\nThreshold: {args.threshold}  |  Correction model: t5-efficient-{args.model}\n")
    scorer_model, scorer_tok = load_scorer()
    corrector = load_corrector(args.model)
    print()

    results = []
    correct_routes = 0
    false_positives = 0  # preserve → routed
    false_negatives = 0  # fix     → not routed

    for text, expected, category in TEST_CASES:
        r = run_pipeline(text, scorer_model, scorer_tok, corrector, args.threshold)

        routed = r["routed"]
        output = r["output"].strip()
        changed = output.lower() != text.lower()

        # Routing correctness
        if category == "fix" and routed:
            route_verdict = "TP"   # correctly sent to corrector
            correct_routes += 1
        elif category == "preserve" and not routed:
            route_verdict = "TN"   # correctly passed through
            correct_routes += 1
        elif category == "preserve" and routed:
            route_verdict = "FP"   # clean text sent to corrector unnecessarily
            false_positives += 1
        else:
            route_verdict = "FN"   # error not caught by scorer
            false_negatives += 1

        results.append({**r, "input": text, "output": output,
                        "category": category, "route_verdict": route_verdict})

    # ── Display ────────────────────────────────────────────────────────────────
    fix_cases     = [r for r in results if r["category"] == "fix"]
    preserve_cases = [r for r in results if r["category"] == "preserve"]

    for group_label, group in [("FIX cases", fix_cases), ("PRESERVE cases", preserve_cases)]:
        print(f"\n── {group_label} " + "─" * 60)
        for r in group:
            routed_tag = "ROUTE" if r["routed"] else "PASS "
            verdict_tag = f"[{r['route_verdict']}]"
            min_tok, min_lp = r["min_token"]
            flagged_str = (
                f"  flagged: {', '.join(repr(t) for t in r['flagged'][:3])}"
                if r["flagged"] else ""
            )

            print(f"\n{verdict_tag} {routed_tag}  {r['input']}")
            print(f"       min-prob token: {repr(min_tok)} ({min_lp:.2f}){flagged_str}")
            if r["routed"]:
                changed_marker = "~" if r["output"].lower() != r["input"].lower() else "="
                print(f"       {changed_marker} corrected: {r['output']}")
            timing = f"score={r['score_ms']:.0f}ms"
            if r["routed"]:
                timing += f"  correct={r['correct_ms']:.0f}ms"
            timing += f"  total={r['total_ms']:.0f}ms"
            print(f"       {timing}")

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(results)
    routed_results = [r for r in results if r["routed"]]
    pass_results   = [r for r in results if not r["routed"]]

    print(f"\n{'─' * 72}")
    print(f"Threshold: {args.threshold}")
    print(f"Routing accuracy: {correct_routes}/{n}  "
          f"TP={sum(1 for r in results if r['route_verdict']=='TP')}  "
          f"TN={sum(1 for r in results if r['route_verdict']=='TN')}  "
          f"FP={false_positives}  FN={false_negatives}")
    print()

    all_total = [r["total_ms"] for r in results]
    pass_total = [r["total_ms"] for r in pass_results]
    route_total = [r["total_ms"] for r in routed_results]

    print(f"Latency — all inputs (n={n}):")
    print(f"  median={statistics.median(all_total):.0f}ms  "
          f"p95={sorted(all_total)[int(n * 0.95)]:.0f}ms  "
          f"max={max(all_total):.0f}ms")
    if pass_results:
        print(f"Latency — passed through (n={len(pass_results)}, scorer only):")
        print(f"  median={statistics.median(pass_total):.0f}ms  "
              f"max={max(pass_total):.0f}ms")
    if routed_results:
        print(f"Latency — routed to corrector (n={len(routed_results)}):")
        print(f"  median={statistics.median(route_total):.0f}ms  "
              f"max={max(route_total):.0f}ms")


if __name__ == "__main__":
    main()
