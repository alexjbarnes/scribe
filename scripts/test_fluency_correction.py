#!/usr/bin/env python3
"""
Evaluate visheratin/t5-efficient-{tiny,mini}-grammar-correction on realistic
ASR dictation errors.

Usage:
    python3 scripts/test_fluency_correction.py              # tiny model
    python3 scripts/test_fluency_correction.py --model mini # mini model
    python3 scripts/test_fluency_correction.py --warmup 3   # more warmup runs

Models are downloaded from HuggingFace on first run (~44MB tiny / ~77MB mini INT8).
Re-runs use the local cache (~/.cache/huggingface).

Requires:
    pip install optimum[onnxruntime] transformers
"""

import argparse
import statistics
import time

# ── Test cases ───────────────────────────────────────────────────────────────
#
# Each tuple is (input, expected_output, category).
# Expected output is what a correct correction would produce.
# The script doesn't enforce expected output — it's for human review.
# Categories: "fix", "preserve", "asr"
#
#   fix      — contains an error that should be corrected
#   preserve — clean text that should NOT be changed
#   asr      — realistic raw ASR output with mixed issues

TEST_CASES = [
    # ── Homophones ──────────────────────────────────────────────────────────
    (
        "your going to love this new feature",
        "you're going to love this new feature",
        "fix / homophone: your→you're",
    ),
    (
        "their is a problem with the build",
        "there is a problem with the build",
        "fix / homophone: their→there",
    ),
    (
        "we need to look at there approach",
        "we need to look at their approach",
        "fix / homophone: there→their",
    ),
    (
        "the server lost its connection",
        "the server lost its connection",
        "preserve / its (possessive) is correct",
    ),
    (
        "its going to take longer than expected",
        "it's going to take longer than expected",
        "fix / homophone: its→it's",
    ),

    # ── Articles ─────────────────────────────────────────────────────────────
    (
        "I want to go to store before the meeting",
        "I want to go to the store before the meeting",
        "fix / missing article: the",
    ),
    (
        "a apple fell from the tree",
        "an apple fell from the tree",
        "fix / a→an before vowel",
    ),
    (
        "we need a update to the documentation",
        "we need an update to the documentation",
        "fix / a→an before vowel",
    ),

    # ── Verb tense ───────────────────────────────────────────────────────────
    (
        "yesterday I remove all the old log files",
        "yesterday I removed all the old log files",
        "fix / tense: remove→removed (past marker present)",
    ),
    (
        "last week we push the release and everything broke",
        "last week we pushed the release and everything broke",
        "fix / tense: push→pushed",
    ),
    (
        "earlier today I fix the bug and deploy it",
        "earlier today I fixed the bug and deployed it",
        "fix / tense: fix→fixed, deploy→deployed",
    ),
    (
        "we remove old logs every Friday",
        "we remove old logs every Friday",
        "preserve / habitual present — should NOT change to removed",
    ),

    # ── Subject-verb agreement ───────────────────────────────────────────────
    (
        "he walk to the office every day",
        "he walks to the office every day",
        "fix / agreement: walk→walks",
    ),
    (
        "the team have completed the project",
        "the team has completed the project",
        "fix / agreement: have→has",
    ),
    (
        "we was looking at the data yesterday",
        "we were looking at the data yesterday",
        "fix / agreement: was→were",
    ),

    # ── Informal / spoken constructions ─────────────────────────────────────
    (
        "I should of called them earlier",
        "I should have called them earlier",
        "fix / should of→should have",
    ),
    (
        "could of been a lot worse",
        "could have been a lot worse",
        "fix / could of→could have",
    ),

    # ── Clean text — should pass through unchanged ───────────────────────────
    (
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",
        "preserve / clean pangram",
    ),
    (
        "I want to schedule a meeting for tomorrow at 3pm.",
        "I want to schedule a meeting for tomorrow at 3pm.",
        "preserve / clean dictation sentence",
    ),
    (
        "We deployed the new API endpoint to production yesterday.",
        "We deployed the new API endpoint to production yesterday.",
        "preserve / technical, already past tense",
    ),
    (
        "Please send me the report by end of day.",
        "Please send me the report by end of day.",
        "preserve / clean imperative",
    ),
    (
        "The function returns a list of strings.",
        "The function returns a list of strings.",
        "preserve / technical, clean",
    ),

    # ── Technical jargon — should not be over-corrected ──────────────────────
    (
        "The PR is in review and CI is green.",
        "The PR is in review and CI is green.",
        "preserve / tech abbreviations",
    ),
    (
        "API endpoints need to be REST-compliant.",
        "API endpoints need to be REST-compliant.",
        "preserve / technical sentence",
    ),

    # ── Realistic raw ASR output ─────────────────────────────────────────────
    (
        "so I was thinking we could look at the the data again",
        "so I was thinking we could look at the data again",
        "asr / duplicate 'the the'",
    ),
    (
        "I I just wanted to say that the deployment went well",
        "I just wanted to say that the deployment went well",
        "asr / duplicate 'I I'",
    ),
    (
        "the meeting is at three thirty on Monday the twenty fourth",
        "the meeting is at three thirty on Monday the twenty fourth",
        "asr / spoken time/date — preserve or normalise",
    ),
    (
        "we discussed the project and then we went to lunch and then we reviewed the pull requests",
        "we discussed the project, then we went to lunch, and then we reviewed the pull requests",
        "asr / run-on sentence — ideally gets punctuation",
    ),
    (
        "can you send that to john at example dot com",
        "can you send that to john at example dot com",
        "asr / spoken email — preserve",
    ),
]


def load_model(model_size: str):
    from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline

    repo = f"visheratin/t5-efficient-{model_size}-grammar-correction"
    print(f"Loading {repo} ...")
    t0 = time.perf_counter()
    model = T5ForConditionalGeneration.from_pretrained(repo)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_ms:.0f}ms\n")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def run_test(corrector, text: str, warmup: int, runs: int) -> tuple[str, float]:
    """Return (corrected_text, median_latency_ms)."""
    times = []
    result = None
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        out = corrector(text, max_length=128)
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(elapsed)
            result = out[0]["generated_text"]
    return result, statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["tiny", "mini"],
        default="tiny",
        help="Model size (default: tiny)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (default: 1)")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per input (default: 3)")
    args = parser.parse_args()

    corrector = load_model(args.model)

    col_w = 46
    header = f"{'INPUT':<{col_w}}  {'OUTPUT':<{col_w}}  {'MS':>6}  CATEGORY"
    print(header)
    print("-" * len(header))

    latencies = []
    changed = 0
    unchanged = 0

    for text, expected, category in TEST_CASES:
        output, ms = run_test(corrector, text, args.warmup, args.runs)
        latencies.append(ms)

        modified = output.strip() != text.strip()
        if modified:
            changed += 1
            marker = "~"
        else:
            unchanged += 1
            marker = " "

        inp = text if len(text) <= col_w else text[: col_w - 1] + "…"
        out = output if len(output) <= col_w else output[: col_w - 1] + "…"
        print(f"{marker} {inp:<{col_w}}  {out:<{col_w}}  {ms:>5.0f}ms  {category}")

    print()
    print(f"Modified: {changed}/{len(TEST_CASES)}  Unchanged: {unchanged}/{len(TEST_CASES)}")
    print(
        f"Latency  min={min(latencies):.0f}ms  "
        f"median={statistics.median(latencies):.0f}ms  "
        f"p95={sorted(latencies)[int(len(latencies) * 0.95)]:.0f}ms  "
        f"max={max(latencies):.0f}ms"
    )


if __name__ == "__main__":
    main()
