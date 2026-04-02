#!/usr/bin/env python3
"""
Full-pipeline test: rule-based stages (nlprule + SymSpell) then neural stage.

Stage 1-4 (Rust):   filler removal → ITN → nlprule grammar → SymSpell spelling
Stage 5   (Python): electra-small router @ 0.75 → Unbabel/gec-t5_small corrector

How it works:
  1. Writes test cases to /tmp/pipeline_input.json
  2. Runs `cargo test --lib pipeline_report -- --nocapture --ignored`
     in src-tauri/ to get rule-based output written to /tmp/pipeline_output.tsv
  3. Reads that TSV and shows what the rule-based stages already fixed
  4. For fix cases still not fixed after rule-based stages, runs the neural stage

Usage:
    python3 scripts/test_full_pipeline.py
    python3 scripts/test_full_pipeline.py --rules-only
    python3 scripts/test_full_pipeline.py --threshold 0.75
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROUTER_MODEL     = "pszemraj/electra-small-discriminator-CoLA"
CORRECTOR_MODEL  = "Unbabel/gec-t5_small"
CORRECTOR_PREFIX = "gec: "
DEFAULT_THRESHOLD = 0.75
REPO_ROOT = Path(__file__).parent.parent
SRC_TAURI = REPO_ROOT / "src-tauri"

# Combined test cases from v2 (developer) and v3 (general public) scripts
TEST_CASES = [
    # ── fix/homophone ──────────────────────────────────────────────────────────
    ("your going to love this new feature",                        "fix/homophone"),
    ("their is a problem with the build",                          "fix/homophone"),
    ("we need to look at there approach",                          "fix/homophone"),
    ("its going to take longer than expected",                     "fix/homophone"),
    ("your going to love this restaurant",                         "fix/homophone"),
    ("their coming over for dinner on Sunday",                     "fix/homophone"),
    ("its going to be a great weekend",                            "fix/homophone"),
    ("make sure your ready before they arrive",                    "fix/homophone"),
    ("I think there going to be late",                             "fix/homophone"),
    ("we need to look at there schedule",                          "fix/homophone"),
    ("the food is to hot to eat right now",                        "fix/homophone"),
    ("we have to many leftovers in the fridge",                    "fix/homophone"),
    ("the store is to far to walk from here",                      "fix/homophone"),
    ("I need to by groceries on the way home",                     "fix/homophone"),
    ("she past the driving test on her first try",                 "fix/homophone"),
    ("the weather is nicer then I expected",                       "fix/homophone"),
    ("he runs faster then anyone on the team",                     "fix/homophone"),
    ("we went to there house for the holidays",                    "fix/homophone"),
    ("he one the championship last year",                          "fix/homophone"),
    ("it has been a hole week since I heard from her",             "fix/homophone"),

    # ── fix/tense ─────────────────────────────────────────────────────────────
    ("yesterday I remove all the old log files",                   "fix/tense"),
    ("last week we push the release and it broke",                 "fix/tense"),
    ("earlier today I fix the bug and deploy it",                  "fix/tense"),
    ("yesterday I call my mom and she sound really tired",         "fix/tense"),
    ("last night I cook a big pasta dinner for everyone",          "fix/tense"),
    ("I wake up late this morning and miss the bus",               "fix/tense"),
    ("she walk into the room and everyone stop talking",           "fix/tense"),
    ("we drive to the beach last Saturday",                        "fix/tense"),
    ("he ask me to help him move last weekend",                    "fix/tense"),
    ("she start a new job at the hospital last month",             "fix/tense"),
    ("the package arrive two days late",                           "fix/tense"),

    # ── fix/agreement ─────────────────────────────────────────────────────────
    ("we was looking at the data yesterday",                       "fix/agreement"),
    ("he walk to the office every day",                            "fix/agreement"),
    ("the function return a list of strings",                      "fix/agreement"),
    ("my sister don't like spicy food",                            "fix/agreement"),
    ("the kids was really excited about the trip",                 "fix/agreement"),
    ("he don't want to go to the party",                           "fix/agreement"),
    ("the store don't open until nine",                            "fix/agreement"),
    ("the movie were really good",                                 "fix/agreement"),
    ("my friends was late to the restaurant",                      "fix/agreement"),
    ("she take the train to work every day",                       "fix/agreement"),
    ("the restaurant close early on Sundays",                      "fix/agreement"),

    # ── fix/informal ──────────────────────────────────────────────────────────
    ("I should of called them earlier",                            "fix/informal"),
    ("could of been a lot worse",                                  "fix/informal"),
    ("we should of tested this more thoroughly",                   "fix/informal"),
    ("I should of brought an umbrella",                            "fix/informal"),
    ("we could of left earlier to avoid traffic",                  "fix/informal"),
    ("she would of loved this restaurant",                         "fix/informal"),
    ("they must of updated the API without telling us",            "fix/informal"),
    ("she must of forgotten about the appointment",                "fix/informal"),
    ("I would of helped if I had known",                           "fix/informal"),
    ("he should of studied harder for the exam",                   "fix/informal"),

    # ── fix/asr-dup ───────────────────────────────────────────────────────────
    ("so I was thinking we could look at the the data again",      "fix/asr-dup"),
    ("I I just wanted to say the deployment went well",            "fix/asr-dup"),
    ("the the problem is in the authentication layer",             "fix/asr-dup"),
    ("so I was saying we could could go out for dinner",           "fix/asr-dup"),
    ("the the concert was amazing last night",                     "fix/asr-dup"),
    ("I need to to make a doctor's appointment",                   "fix/asr-dup"),
    ("the the kids are already in bed",                            "fix/asr-dup"),
    ("we need need to leave in about ten minutes",                 "fix/asr-dup"),

    # ── fix/article ───────────────────────────────────────────────────────────
    ("I want to go to store before the meeting",                   "fix/article"),
    ("we need a update to the documentation",                      "fix/article"),
    ("I need to make appointment with the dentist",                "fix/article"),
    ("we had wonderful time at the wedding",                       "fix/article"),
    ("she is nurse at the local hospital",                         "fix/article"),
    ("can you make reservation for tomorrow night",                "fix/article"),
    ("I have important meeting this afternoon",                    "fix/article"),
    ("she is wearing a amazing dress",                             "fix/article"),
    ("I took a Uber to the airport",                               "fix/article"),
    ("it was a honor to meet her",                                 "fix/article"),

    # ── fix/plural ────────────────────────────────────────────────────────────
    ("we have three server running in production",                 "fix/plural"),
    ("I bought three new shirt for the trip",                      "fix/plural"),
    ("she has two dog and one cat",                                "fix/plural"),
    ("there are three doctor in this practice",                    "fix/plural"),
    ("he has been to many different country",                      "fix/plural"),

    # ── fix/contraction ───────────────────────────────────────────────────────
    ("I cant believe how fast the year went by",                   "fix/contraction"),
    ("she doesnt want to come to the party",                       "fix/contraction"),
    ("we wont be home until late tonight",                         "fix/contraction"),
    ("he didnt tell me about the change of plans",                 "fix/contraction"),
    ("I wasnt expecting that kind of response",                    "fix/contraction"),
    ("they arent coming to the dinner tonight",                    "fix/contraction"),
    ("I havent been to that restaurant yet",                       "fix/contraction"),
    ("she wouldnt take no for an answer",                          "fix/contraction"),
    ("I couldnt find parking near the venue",                      "fix/contraction"),
    ("they dont know what they are missing",                       "fix/contraction"),

    # ── preserve/everyday ─────────────────────────────────────────────────────
    ("I need to pick up the kids from school at three.",           "preserve/everyday"),
    ("We are having dinner at my parents' house tonight.",         "preserve/everyday"),
    ("Can you remind me to call the dentist tomorrow?",            "preserve/everyday"),
    ("I am going to the gym after work today.",                    "preserve/everyday"),
    ("I think I left my phone in the car.",                        "preserve/everyday"),
    ("The kids have a school play this Friday evening.",           "preserve/everyday"),
    ("She got a new haircut and it looks great.",                  "preserve/everyday"),
    ("We had a great time at the concert last night.",             "preserve/everyday"),
    ("The flight is at seven in the morning.",                     "preserve/everyday"),
    ("I have a dentist appointment on Wednesday morning.",         "preserve/everyday"),

    # ── preserve/tech ─────────────────────────────────────────────────────────
    ("We deployed the new API endpoint yesterday.",                "preserve/tech"),
    ("The function returns a list of strings.",                    "preserve/tech"),
    ("API endpoints need to be REST-compliant.",                   "preserve/tech"),
    ("The PR is in review and CI is green.",                       "preserve/tech"),
    ("The pipeline runs on every push to main.",                   "preserve/tech"),
    ("I added a GitHub Actions workflow for linting.",             "preserve/tech"),
    ("The INT8 quantized encoder is thirty-four megabytes.",       "preserve/tech"),
    ("The model runs on ARM64 with NEON optimizations.",           "preserve/tech"),
    ("I updated the Cargo.toml to bump the sherpa dependency.",    "preserve/tech"),
    ("We shard the database by user ID for horizontal scaling.",   "preserve/tech"),

    # ── preserve/proper-nouns ─────────────────────────────────────────────────
    ("I need to pick up my order from Amazon.",                    "preserve/proper-nouns"),
    ("I ran into Sarah at the grocery store yesterday.",           "preserve/proper-nouns"),
    ("We are flying to Miami for the long weekend.",               "preserve/proper-nouns"),
    ("I bought this at Target last week.",                         "preserve/proper-nouns"),
    ("I ordered a burrito from Chipotle for lunch.",               "preserve/proper-nouns"),
    ("I got a gift card to Starbucks for my birthday.",            "preserve/proper-nouns"),
    ("I called my mom on FaceTime last night.",                    "preserve/proper-nouns"),
    ("The Lakers won by twelve points last night.",                "preserve/proper-nouns"),
    ("He drives a Toyota Camry.",                                  "preserve/proper-nouns"),
    ("I use Spotify for music and Netflix for shows.",             "preserve/proper-nouns"),

    # ── preserve/habitual ─────────────────────────────────────────────────────
    ("we remove old logs every Friday",                            "preserve/habitual"),
    ("I walk the dog every morning before breakfast.",             "preserve/habitual"),
    ("She calls her mom every Sunday afternoon.",                  "preserve/habitual"),
    ("We eat dinner as a family every night.",                     "preserve/habitual"),
    ("the team deploys every two weeks",                           "preserve/habitual"),
    ("I review PRs every morning before standup",                  "preserve/habitual"),
    ("the garbage collector runs every thirty seconds",            "preserve/habitual"),
    ("He goes to the gym three times a week.",                     "preserve/habitual"),

    # ── preserve/formal ───────────────────────────────────────────────────────
    ("Please confirm your attendance by Friday.",                  "preserve/formal"),
    ("Your application has been received and is under review.",    "preserve/formal"),
    ("We regret to inform you that the position has been filled.", "preserve/formal"),
    ("We will be in touch once a decision has been made.",         "preserve/formal"),
    ("Please review the attached document before the meeting.",    "preserve/formal"),
]


def normalize(s: str) -> str:
    return s.strip().rstrip(".!?").strip().lower()


def p_acceptable(result: dict) -> float:
    label = result["label"].lower()
    score = result["score"]
    return 1.0 - score if "unacceptable" in label else score


def run_rust_pipeline():
    print("Writing test cases to /tmp/pipeline_input.json ...")
    with open("/tmp/pipeline_input.json", "w") as f:
        json.dump([(text, cat) for text, cat in TEST_CASES], f)

    print("Running Rust pipeline stages (nlprule + SymSpell) ...")
    t0 = time.perf_counter()
    r = subprocess.run(
        ["cargo", "test", "--lib", "pipeline_report", "--", "--nocapture", "--ignored"],
        cwd=SRC_TAURI,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print("cargo test failed:", r.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    print(f"  Rust pipeline done in {elapsed:.1f}s")

    rows = []
    with open("/tmp/pipeline_output.tsv") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                rows.append({
                    "input": parts[0],
                    "rules_output": parts[1],
                    "rules_changed": parts[2] == "true",
                    "category": parts[3],
                })
    return rows


def print_rules_summary(rows):
    fix_cases      = [r for r in rows if r["category"].startswith("fix")]
    preserve_cases = [r for r in rows if r["category"].startswith("preserve")]

    rules_fixed    = sum(1 for r in fix_cases      if r["rules_changed"])
    rules_corrupt  = sum(1 for r in preserve_cases if r["rules_changed"])

    print(f"\n── Rule-based stages (nlprule + SymSpell) ──────────────────────────────")
    print(f"  fix cases:      {rules_fixed}/{len(fix_cases)} changed by rules")
    print(f"  preserve cases: {rules_corrupt}/{len(preserve_cases)} changed by rules (FP)")

    cats = sorted({r["category"] for r in rows})
    print(f"\n  {'Category':<26}  total  fixed/flagged")
    for cat in cats:
        group   = [r for r in rows if r["category"] == cat]
        changed = sum(1 for r in group if r["rules_changed"])
        print(f"  {cat:<26}  {len(group):5}  {changed}")

    corruptions = [r for r in preserve_cases if r["rules_changed"]]
    if corruptions:
        print(f"\n  ── Rule-based corruptions (FP={len(corruptions)}) ──")
        for r in corruptions:
            print(f"    [{r['category']}]  {r['input']}")
            print(f"              → {r['rules_output']}")

    # Show what fix cases the rules already handled
    rules_fixed_list = [r for r in fix_cases if r["rules_changed"]]
    if rules_fixed_list:
        col = 46
        print(f"\n  ── Fixed by rules ──")
        for r in rules_fixed_list:
            inp = r["input"][:col] + "…" if len(r["input"]) > col else r["input"]
            out = r["rules_output"][:col] + "…" if len(r["rules_output"]) > col else r["rules_output"]
            print(f"    [{r['category']}]  {inp:<{col}}  →  {out}")


def run_neural_stage(rows, threshold, rules_only):
    # Neural stage runs on the rules_output (what's left after rule-based stages)
    # For fix cases not yet fixed, router+corrector gets a shot
    # For all preserve cases (including those rules already handled), we still run
    # router to confirm no FPs
    if rules_only:
        return rows

    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"\nLoading router: {ROUTER_MODEL} ...")
    t0 = time.perf_counter()
    router = hf_pipeline("text-classification", model=ROUTER_MODEL)
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    print(f"\nScoring {len(rows)} sentences through neural router (threshold={threshold}) ...")
    for i, r in enumerate(rows):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(rows)} ...")
        t0 = time.perf_counter()
        raw = router(r["rules_output"])[0]
        r["router_ms"] = (time.perf_counter() - t0) * 1000
        r["pa"] = p_acceptable(raw)
        r["neural_route"] = r["pa"] < threshold

    routed = [r for r in rows if r["neural_route"]]
    if not routed:
        print("  No sentences routed to corrector.")
        return rows

    print(f"\nLoading corrector: {CORRECTOR_MODEL} ...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(CORRECTOR_MODEL)
    mod = AutoModelForSeq2SeqLM.from_pretrained(CORRECTOR_MODEL)
    corrector = hf_pipeline("text2text-generation", model=mod, tokenizer=tok)
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    print(f"\nRunning corrector on {len(routed)} routed sentences ...")
    for i, r in enumerate(routed):
        if i % 20 == 0 and i > 0:
            print(f"  {i}/{len(routed)} ...")
        t0 = time.perf_counter()
        r["neural_output"] = corrector(
            CORRECTOR_PREFIX + r["rules_output"], max_new_tokens=128
        )[0]["generated_text"].strip()
        r["corrector_ms"] = (time.perf_counter() - t0) * 1000

    return rows


def print_full_summary(rows, threshold):
    fix_cases      = [r for r in rows if r["category"].startswith("fix")]
    preserve_cases = [r for r in rows if r["category"].startswith("preserve")]

    def final_output(r):
        return r.get("neural_output", r["rules_output"])

    def is_fixed(r):
        return normalize(final_output(r)) != normalize(r["input"])

    def is_fully_correct(r):
        # For fix cases: was something changed (either by rules or neural)?
        return r["rules_changed"] or r.get("neural_route", False)

    rules_fixed   = [r for r in fix_cases if r["rules_changed"]]
    neural_fixed  = [r for r in fix_cases if not r["rules_changed"] and r.get("neural_route", False)]
    still_missed  = [r for r in fix_cases if not r["rules_changed"] and not r.get("neural_route", False)]
    rules_corrupt = [r for r in preserve_cases if r["rules_changed"]]
    neural_fp     = [r for r in preserve_cases if not r["rules_changed"] and r.get("neural_route", False)]

    n = len(rows)
    print(f"\n{'═'*76}")
    print(f"  FULL PIPELINE SUMMARY  ({n} sentences)")
    print(f"{'═'*76}")
    print(f"  fix cases:      {len(fix_cases)}")
    print(f"    Fixed by rules only:         {len(rules_fixed)}")
    print(f"    Routed to neural (not fixed by rules): {len(neural_fixed)}")
    print(f"    Still missed (FN):           {len(still_missed)}")
    print(f"  preserve cases: {len(preserve_cases)}")
    print(f"    Rules FP (changed clean text): {len(rules_corrupt)}")
    print(f"    Neural FP (routed clean text): {len(neural_fp)}")

    col = 50
    if neural_fixed:
        print(f"\n  ── Neural stage fixed (after rules passed) ──")
        for r in neural_fixed:
            inp = r["rules_output"][:col] + "…" if len(r["rules_output"]) > col else r["rules_output"]
            out = r.get("neural_output", "")[:col] + "…" if len(r.get("neural_output", "")) > col else r.get("neural_output", "")
            print(f"    [{r['category']}]  {inp:<{col}}  →  {out}")

    if still_missed:
        print(f"\n  ── Still missed after full pipeline (FN={len(still_missed)}) ──")
        cats = {}
        for r in still_missed:
            cats.setdefault(r["category"], []).append(r)
        for cat in sorted(cats):
            print(f"    {cat}: {len(cats[cat])} cases")
            for r in cats[cat]:
                print(f"      {r['input']}")

    if neural_fp:
        print(f"\n  ── Neural FP (clean text incorrectly routed, p={threshold}) ──")
        for r in neural_fp:
            inp = r["rules_output"][:col] + "…" if len(r["rules_output"]) > col else r["rules_output"]
            out = r.get("neural_output", "")[:col] + "…" if len(r.get("neural_output", "")) > col else r.get("neural_output", "")
            changed = normalize(r.get("neural_output", "")) != normalize(r["rules_output"])
            mark = "!" if changed else "="
            print(f"  {mark} [{r['category']}]  p={r['pa']:.3f}  {inp}")
            if changed:
                print(f"      → {out}")

    # Per-category breakdown
    cats = sorted({r["category"] for r in rows})
    print(f"\n  Per-category (full pipeline):")
    print(f"  {'Category':<26}  total  fixed/preserved  missed/FP")
    for cat in cats:
        group = [r for r in rows if r["category"] == cat]
        if cat.startswith("fix"):
            fixed  = sum(1 for r in group if r["rules_changed"] or r.get("neural_route", False))
            missed = len(group) - fixed
            print(f"  {cat:<26}  {len(group):5}  {fixed:15}  {missed}")
        else:
            total_fp = sum(1 for r in group if r["rules_changed"] or r.get("neural_route", False))
            print(f"  {cat:<26}  {len(group):5}  {len(group)-total_fp:15}  FP={total_fp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--rules-only", action="store_true",
                        help="Show rule-based results only, skip neural stage")
    args = parser.parse_args()

    rows = run_rust_pipeline()
    print_rules_summary(rows)

    rows = run_neural_stage(rows, args.threshold, args.rules_only)

    if not args.rules_only:
        print_full_summary(rows, args.threshold)


if __name__ == "__main__":
    main()
