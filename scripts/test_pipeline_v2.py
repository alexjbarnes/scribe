#!/usr/bin/env python3
"""
End-to-end two-stage pipeline test: 220 cases.

Stage 1: pszemraj/electra-small-discriminator-CoLA at threshold=0.75
Stage 2: Unbabel/gec-t5_small (only for sentences routed by stage 1)

Categories:
  fix/homophone   - your/you're, their/there, its/it's, to/too, then/than, etc.
  fix/article     - missing or wrong a/an/the
  fix/tense       - wrong verb tense (past narration with present form)
  fix/agreement   - subject-verb disagreement
  fix/informal    - "of" for "have" (should of, could of, etc.)
  fix/asr-dup     - ASR word repetitions
  fix/plural      - wrong singular/plural form
  preserve/clean  - clean everyday sentences
  preserve/tech   - technical developer sentences with jargon
  preserve/habitual - habitual present tense (valid grammar, looks like tense errors)
  preserve/formal - formal business prose

Usage:
    python3 scripts/test_pipeline_v2.py
    python3 scripts/test_pipeline_v2.py --threshold 0.80
    python3 scripts/test_pipeline_v2.py --router-only
    python3 scripts/test_pipeline_v2.py --runs 2
"""

import argparse
import statistics
import time

ROUTER_MODEL     = "pszemraj/electra-small-discriminator-CoLA"
CORRECTOR_MODEL  = "Unbabel/gec-t5_small"
CORRECTOR_PREFIX = "gec: "
DEFAULT_THRESHOLD = 0.75

TEST_CASES = [
    # ── fix/homophone ──────────────────────────────────────────────────────────
    ("your going to love this new feature",                         "fix/homophone"),
    ("their is a problem with the build",                           "fix/homophone"),
    ("we need to look at there approach",                           "fix/homophone"),
    ("its going to take longer than expected",                      "fix/homophone"),
    ("they said there ready to deploy",                             "fix/homophone"),
    ("your the one who approved the PR",                            "fix/homophone"),
    ("make sure its running before you leave",                      "fix/homophone"),
    ("there going to need more time on this",                       "fix/homophone"),
    ("its not going to fit in the buffer",                          "fix/homophone"),
    ("I think there approach is wrong",                             "fix/homophone"),
    ("your changes broke the pipeline",                             "fix/homophone"),
    ("the team said there ready for the review",                    "fix/homophone"),
    ("its clear that the bug is in the parser",                     "fix/homophone"),
    ("we know there going to miss the deadline",                    "fix/homophone"),
    ("the files are to large to upload",                            "fix/homophone"),
    ("its to early to tell if the fix worked",                      "fix/homophone"),
    ("the response time is to slow for production",                 "fix/homophone"),
    ("we have to many open issues right now",                       "fix/homophone"),
    ("the buffer is to small for this payload",                     "fix/homophone"),
    ("I want to no if the tests are passing",                       "fix/homophone"),
    ("the whole system went down witch was unexpected",             "fix/homophone"),
    ("we past the deadline by two weeks",                           "fix/homophone"),
    ("the accept rate is higher then expected",                     "fix/homophone"),
    ("the new version is more stable then the old one",             "fix/homophone"),
    ("this approach is better then what we had before",             "fix/homophone"),

    # ── fix/article ───────────────────────────────────────────────────────────
    ("I want to go to store before the meeting",                    "fix/article"),
    ("we need a update to the documentation",                       "fix/article"),
    ("can you send me link to the dashboard",                       "fix/article"),
    ("I opened issue in the tracker",                               "fix/article"),
    ("we pushed update to the main branch",                         "fix/article"),
    ("there is error in the logs",                                  "fix/article"),
    ("I need access to database",                                   "fix/article"),
    ("I wrote unit test for the function",                          "fix/article"),
    ("can you create pull request for this",                        "fix/article"),
    ("there was outage last night",                                 "fix/article"),
    ("we have meeting at three o'clock",                            "fix/article"),
    ("I need to open ticket for this bug",                          "fix/article"),
    ("its an useful feature to have",                               "fix/article"),
    ("we added a endpoint for file uploads",                        "fix/article"),
    ("I wrote a overview of the changes",                           "fix/article"),
    ("she gave an useful summary of the results",                   "fix/article"),
    ("we noticed a unusual pattern in the logs",                    "fix/article"),
    ("I have a hour before the meeting",                            "fix/article"),
    ("she is engineer on the team",                                 "fix/article"),
    ("the server gave a error message",                             "fix/article"),

    # ── fix/tense ─────────────────────────────────────────────────────────────
    ("yesterday I remove all the old log files",                    "fix/tense"),
    ("last week we push the release and it broke",                  "fix/tense"),
    ("earlier today I fix the bug and deploy it",                   "fix/tense"),
    ("last month we migrate the database",                          "fix/tense"),
    ("yesterday the team finish the sprint",                        "fix/tense"),
    ("I already send the email this morning",                       "fix/tense"),
    ("she mention the issue in the standup",                        "fix/tense"),
    ("we complete the migration last Tuesday",                      "fix/tense"),
    ("I notice the problem an hour ago",                            "fix/tense"),
    ("the build fail three times yesterday",                        "fix/tense"),
    ("last week I add the new configuration option",                "fix/tense"),
    ("we review the PR and merge it yesterday",                     "fix/tense"),
    ("the team decide to roll back the deployment",                 "fix/tense"),
    ("yesterday we discover a memory leak in the service",          "fix/tense"),
    ("I submit the form but nothing happen",                        "fix/tense"),
    ("she create a new branch for the hotfix",                      "fix/tense"),
    ("we open the investigation last Friday",                       "fix/tense"),
    ("the server crash during the deploy last night",               "fix/tense"),
    ("I check the logs and find the error",                         "fix/tense"),
    ("last quarter we reduce the latency by forty percent",         "fix/tense"),
    ("the engineer fix the issue without any help",                 "fix/tense"),
    ("we realize the problem after the deployment",                 "fix/tense"),
    ("I spend the whole morning debugging this",                    "fix/tense"),
    ("the team agree to the new approach last week",                "fix/tense"),
    ("she push the hotfix before the meeting",                      "fix/tense"),

    # ── fix/agreement ─────────────────────────────────────────────────────────
    ("we was looking at the data yesterday",                        "fix/agreement"),
    ("the results was disappointing",                               "fix/agreement"),
    ("the errors was caused by a race condition",                   "fix/agreement"),
    ("the files was uploaded successfully",                         "fix/agreement"),
    ("he walk to the office every day",                             "fix/agreement"),
    ("she work on the backend team",                                "fix/agreement"),
    ("the server run on port 8080",                                 "fix/agreement"),
    ("the function return a list of strings",                       "fix/agreement"),
    ("the script run every hour",                                   "fix/agreement"),
    ("this approach work well in practice",                         "fix/agreement"),
    ("the database contain millions of records",                    "fix/agreement"),
    ("every endpoint require authentication",                       "fix/agreement"),
    ("the team meet every Monday morning",                          "fix/agreement"),
    ("the API accept JSON and XML payloads",                        "fix/agreement"),
    ("the worker process each job in the queue",                    "fix/agreement"),
    ("the library expose a clean interface",                        "fix/agreement"),
    ("the container start automatically after a crash",             "fix/agreement"),
    ("each microservice handle its own database",                   "fix/agreement"),
    ("the pipeline validate the schema before ingestion",           "fix/agreement"),
    ("the health check endpoint return a status code",              "fix/agreement"),

    # ── fix/informal ──────────────────────────────────────────────────────────
    ("I should of called them earlier",                             "fix/informal"),
    ("could of been a lot worse",                                   "fix/informal"),
    ("we should of tested this more thoroughly",                    "fix/informal"),
    ("they would of caught this in review",                         "fix/informal"),
    ("I could of sworn the test was passing",                       "fix/informal"),
    ("we might of missed something in the requirements",            "fix/informal"),
    ("he should of read the documentation first",                   "fix/informal"),
    ("they must of updated the API without telling us",             "fix/informal"),
    ("I would of fixed it earlier if I knew",                       "fix/informal"),
    ("we could of avoided this entire incident",                    "fix/informal"),
    ("she must of forgotten to run the tests",                      "fix/informal"),
    ("I would of approved the PR earlier",                          "fix/informal"),
    ("the team could of shipped this last week",                    "fix/informal"),
    ("you should of mentioned that in the standup",                 "fix/informal"),
    ("we might of introduced a regression here",                    "fix/informal"),

    # ── fix/asr-dup ───────────────────────────────────────────────────────────
    ("so I was thinking we could look at the the data again",       "fix/asr-dup"),
    ("I I just wanted to say the deployment went well",             "fix/asr-dup"),
    ("can we we schedule a call for tomorrow",                      "fix/asr-dup"),
    ("the the problem is in the authentication layer",              "fix/asr-dup"),
    ("please review the the pull request when you have time",       "fix/asr-dup"),
    ("I need to to check the configuration",                        "fix/asr-dup"),
    ("we should should document this decision",                     "fix/asr-dup"),
    ("let me me explain the issue",                                 "fix/asr-dup"),
    ("the error the error appears in the console",                  "fix/asr-dup"),
    ("I want to to make sure we handle edge cases",                 "fix/asr-dup"),
    ("we we need to update the documentation",                      "fix/asr-dup"),
    ("can you you review my changes",                               "fix/asr-dup"),
    ("the the test suite is taking too long",                       "fix/asr-dup"),
    ("I I think we need to refactor this module",                   "fix/asr-dup"),
    ("please please make sure the deploy is smoke tested",          "fix/asr-dup"),

    # ── fix/plural ────────────────────────────────────────────────────────────
    ("we have three server running in production",                  "fix/plural"),
    ("the two endpoint need to be updated",                         "fix/plural"),
    ("she submitted four pull request today",                       "fix/plural"),
    ("we found several issue in the codebase",                      "fix/plural"),
    ("the test suite has fifty test",                               "fix/plural"),
    ("we are running two instance of the service",                  "fix/plural"),
    ("the team opened five ticket last week",                       "fix/plural"),
    ("there are three open PR that need review",                    "fix/plural"),
    ("we have two database replica in each region",                 "fix/plural"),
    ("the system sends notification to all subscriber",             "fix/plural"),

    # ── preserve/clean ────────────────────────────────────────────────────────
    ("The quick brown fox jumps over the lazy dog.",               "preserve/clean"),
    ("I want to schedule a meeting for tomorrow.",                 "preserve/clean"),
    ("Please send me the report by end of day.",                   "preserve/clean"),
    ("Can you review my code when you get a chance?",              "preserve/clean"),
    ("I think we should discuss this before merging.",             "preserve/clean"),
    ("Let me know if you have any questions.",                     "preserve/clean"),
    ("The meeting has been rescheduled to Thursday.",              "preserve/clean"),
    ("I will look into this and get back to you.",                 "preserve/clean"),
    ("Thanks for catching that bug.",                              "preserve/clean"),
    ("We need to talk about the project timeline.",                "preserve/clean"),
    ("I pushed the changes to the feature branch.",               "preserve/clean"),
    ("Could you take a look at the failing tests?",               "preserve/clean"),
    ("The documentation has been updated.",                        "preserve/clean"),
    ("I finished the review and left some comments.",             "preserve/clean"),
    ("Let's sync up after the standup.",                           "preserve/clean"),
    ("I am working on the authentication flow right now.",        "preserve/clean"),
    ("The staging environment is back online.",                   "preserve/clean"),
    ("I think the issue is in the error handling.",               "preserve/clean"),
    ("We should write more tests for this module.",               "preserve/clean"),
    ("The release is scheduled for next Friday.",                 "preserve/clean"),
    ("I need to update the dependencies.",                        "preserve/clean"),
    ("Can we move the meeting to Friday?",                        "preserve/clean"),
    ("I found the root cause of the memory leak.",                "preserve/clean"),
    ("I merged the hotfix into production.",                      "preserve/clean"),
    ("The team is making good progress on the migration.",        "preserve/clean"),
    ("I reviewed the proposal and have some concerns.",           "preserve/clean"),
    ("Could you send me the meeting invite?",                     "preserve/clean"),
    ("The rollback went smoothly and the service is stable.",     "preserve/clean"),
    ("I spent the afternoon debugging the network issue.",        "preserve/clean"),
    ("The code review is blocking the deployment.",               "preserve/clean"),

    # ── preserve/tech ─────────────────────────────────────────────────────────
    ("We deployed the new API endpoint yesterday.",               "preserve/tech"),
    ("The function returns a list of strings.",                   "preserve/tech"),
    ("API endpoints need to be REST-compliant.",                  "preserve/tech"),
    ("The PR is in review and CI is green.",                      "preserve/tech"),
    ("The service is running on Kubernetes in us-east-1.",       "preserve/tech"),
    ("We use Redis for session caching.",                         "preserve/tech"),
    ("The gRPC service handles about ten thousand RPS.",         "preserve/tech"),
    ("The JWT token expires after twenty-four hours.",           "preserve/tech"),
    ("We have Prometheus metrics for all the endpoints.",        "preserve/tech"),
    ("The pipeline runs on every push to main.",                 "preserve/tech"),
    ("I updated the Dockerfile to use the slim base image.",     "preserve/tech"),
    ("The ORM generates the SQL at runtime.",                    "preserve/tech"),
    ("We use feature flags for gradual rollouts.",               "preserve/tech"),
    ("The ONNX runtime runs the model in under fifty ms.",       "preserve/tech"),
    ("The Android APK is around fifty megabytes.",               "preserve/tech"),
    ("I added a GitHub Actions workflow for linting.",           "preserve/tech"),
    ("The webhook fires on every merged PR.",                    "preserve/tech"),
    ("We use semantic versioning for the SDK.",                  "preserve/tech"),
    ("The load balancer distributes traffic across four pods.",  "preserve/tech"),
    ("The INT8 quantized encoder is thirty-four megabytes.",     "preserve/tech"),
    ("The CI pipeline failed on the integration tests.",         "preserve/tech"),
    ("I added retry logic for transient network errors.",        "preserve/tech"),
    ("The SQLite WAL mode improves concurrent reads.",           "preserve/tech"),
    ("The model runs on ARM64 with NEON optimizations.",         "preserve/tech"),
    ("I updated the Cargo.toml to bump the sherpa dependency.",  "preserve/tech"),
    ("The health check endpoint returns a two hundred response.","preserve/tech"),
    ("We shard the database by user ID for horizontal scaling.", "preserve/tech"),
    ("The token refresh happens automatically before expiry.",   "preserve/tech"),
    ("I configured nginx as a reverse proxy for WebSocket.",     "preserve/tech"),
    ("The test mocks the HTTP client to avoid network calls.",   "preserve/tech"),

    # ── preserve/habitual ─────────────────────────────────────────────────────
    ("we remove old logs every Friday",                          "preserve/habitual"),
    ("the team deploys every two weeks",                         "preserve/habitual"),
    ("I review PRs every morning before standup",                "preserve/habitual"),
    ("we rotate API keys every ninety days",                     "preserve/habitual"),
    ("the backup job runs at midnight every day",                "preserve/habitual"),
    ("I check the metrics dashboard every morning",              "preserve/habitual"),
    ("we run load tests before every major release",             "preserve/habitual"),
    ("the team syncs every Monday and Wednesday",                "preserve/habitual"),
    ("I update the dependencies every sprint",                   "preserve/habitual"),
    ("the monitoring system alerts on every anomaly",            "preserve/habitual"),
    ("we prune old branches every month",                        "preserve/habitual"),
    ("the garbage collector runs every thirty seconds",          "preserve/habitual"),
    ("I archive old tickets every quarter",                      "preserve/habitual"),
    ("we publish release notes every two weeks",                 "preserve/habitual"),
    ("the cron job cleans up temp files every hour",             "preserve/habitual"),

    # ── preserve/formal ───────────────────────────────────────────────────────
    ("The engineering team has made significant progress on the migration.", "preserve/formal"),
    ("Please ensure all tests pass before requesting a review.", "preserve/formal"),
    ("The incident was caused by a misconfigured load balancer.", "preserve/formal"),
    ("We are targeting a release at the end of Q2.",            "preserve/formal"),
    ("The proposal has been reviewed and approved by the architecture team.", "preserve/formal"),
    ("All changes must be reviewed by at least two engineers.", "preserve/formal"),
    ("The performance regression was introduced in the last deployment.", "preserve/formal"),
    ("We have escalated the issue to the on-call team.",        "preserve/formal"),
    ("The migration is expected to complete by the end of the week.", "preserve/formal"),
    ("Please document any breaking changes in the release notes.", "preserve/formal"),
    ("The security audit identified three low-severity findings.", "preserve/formal"),
    ("We will coordinate the rollout with the infrastructure team.", "preserve/formal"),
    ("The root cause analysis has been shared with all stakeholders.", "preserve/formal"),
    ("I have verified the fix against all known failure scenarios.", "preserve/formal"),
    ("The service level agreement requires ninety-nine point nine percent uptime.", "preserve/formal"),
]


def p_acceptable(result: dict) -> float:
    label = result["label"].lower()
    score = result["score"]
    return 1.0 - score if "unacceptable" in label else score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--router-only", action="store_true",
                        help="Skip corrector, show router stats only")
    parser.add_argument("--runs", type=int, default=1,
                        help="Corrector passes per sentence for latency averaging")
    args = parser.parse_args()

    from transformers import pipeline as hf_pipeline

    print(f"\nLoading router: {ROUTER_MODEL} ...")
    t0 = time.perf_counter()
    router = hf_pipeline("text-classification", model=ROUTER_MODEL)
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    # Score all sentences through the router
    n = len(TEST_CASES)
    print(f"\nScoring {n} sentences (threshold={args.threshold}) ...")
    router_results = []
    for i, (text, category) in enumerate(TEST_CASES):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{n} ...")
        t0 = time.perf_counter()
        raw = router(text)[0]
        ms = (time.perf_counter() - t0) * 1000
        pa = p_acceptable(raw)
        router_results.append({
            "text": text,
            "category": category,
            "pa": pa,
            "route": pa < args.threshold,
            "router_ms": ms,
        })

    # Router accuracy metrics
    fix_cases      = [r for r in router_results if r["category"].startswith("fix")]
    preserve_cases = [r for r in router_results if r["category"].startswith("preserve")]
    tp = sum(1 for r in fix_cases      if r["route"])
    fn = sum(1 for r in fix_cases      if not r["route"])
    fp = sum(1 for r in preserve_cases if r["route"])
    tn = sum(1 for r in preserve_cases if not r["route"])
    acc  = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0
    router_ms = [r["router_ms"] for r in router_results]
    routed    = [r for r in router_results if r["route"]]

    print(f"\n── Router results ({n} sentences) ──────────────────────────────────────")
    print(f"  fix={len(fix_cases)}  preserve={len(preserve_cases)}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Acc={acc:.0%}  Prec={prec:.0%}  Recall={rec:.0%}  F1={f1:.2f}")
    print(f"  Routed to corrector: {len(routed)}/{n}")
    print(f"  Router latency: median={statistics.median(router_ms):.1f}ms  "
          f"p95={sorted(router_ms)[int(n*0.95)]:.1f}ms")

    # Per-category routing breakdown
    cats = sorted({r["category"] for r in router_results})
    print(f"\n  Per-category routing:")
    print(f"  {'Category':<22}  total  routed  FP/FN")
    for cat in cats:
        group   = [r for r in router_results if r["category"] == cat]
        routed_g = sum(1 for r in group if r["route"])
        if cat.startswith("fix"):
            miss = sum(1 for r in group if not r["route"])
            print(f"  {cat:<22}  {len(group):5}  {routed_g:6}  FN={miss}")
        else:
            false_pos = sum(1 for r in group if r["route"])
            print(f"  {cat:<22}  {len(group):5}  {routed_g:6}  FP={false_pos}")

    # False negatives and positives detail
    fn_cases = [r for r in fix_cases      if not r["route"]]
    fp_cases = [r for r in preserve_cases if r["route"]]
    if fn_cases:
        print(f"\n  ── False Negatives (FN={len(fn_cases)}) — not routed ──")
        for r in sorted(fn_cases, key=lambda x: x["pa"], reverse=True):
            print(f"    p={r['pa']:.3f}  [{r['category']}]  {r['text']}")
    if fp_cases:
        print(f"\n  ── False Positives (FP={len(fp_cases)}) — incorrectly routed ──")
        for r in sorted(fp_cases, key=lambda x: x["pa"]):
            print(f"    p={r['pa']:.3f}  [{r['category']}]  {r['text']}")

    if args.router_only or not routed:
        return

    # Load corrector
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"\nLoading corrector: {CORRECTOR_MODEL} ...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(CORRECTOR_MODEL)
    mod = AutoModelForSeq2SeqLM.from_pretrained(CORRECTOR_MODEL)
    corrector = hf_pipeline("text2text-generation", model=mod, tokenizer=tok)
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    # Run corrector on routed sentences
    total = len(routed)
    print(f"\nRunning corrector on {total} routed sentences ...")
    for i, r in enumerate(routed):
        if i % 20 == 0 and i > 0:
            print(f"  {i}/{total} ...")
        times = []
        out = None
        for _ in range(args.runs):
            t0 = time.perf_counter()
            out = corrector(
                CORRECTOR_PREFIX + r["text"], max_new_tokens=128
            )[0]["generated_text"].strip()
            times.append((time.perf_counter() - t0) * 1000)
        r["corrected"] = out
        r["corrector_ms"] = statistics.median(times)
        r["changed"] = out.lower().strip() != r["text"].lower().strip()

    routed_fix      = [r for r in routed if r["category"].startswith("fix")]
    routed_preserve = [r for r in routed if r["category"].startswith("preserve")]

    col = 50
    if routed_fix:
        print(f"\n── Corrector on TP cases (fix sentences routed correctly) ──────────────")
        for r in routed_fix:
            marker = "~" if r["changed"] else "="
            inp = r["text"][:col] + "…" if len(r["text"]) > col else r["text"]
            out = r["corrected"][:col] + "…" if len(r["corrected"]) > col else r["corrected"]
            print(f"  {marker} {inp:<{col}}  →  {out:<{col}}  {r['corrector_ms']:.0f}ms  [{r['category']}]")

    if routed_preserve:
        print(f"\n── Corrector on FP cases (preserve sentences incorrectly routed) ───────")
        for r in routed_preserve:
            marker = "!" if r["changed"] else "="
            inp = r["text"][:col] + "…" if len(r["text"]) > col else r["text"]
            out = r["corrected"][:col] + "…" if len(r["corrected"]) > col else r["corrected"]
            print(f"  {marker} {inp:<{col}}  →  {out:<{col}}  {r['corrector_ms']:.0f}ms  [{r['category']}]")

    corrector_changed_tp  = sum(1 for r in routed_fix      if r["changed"])
    corrector_changed_fp  = sum(1 for r in routed_preserve if r["changed"])
    corrector_ms          = [r["corrector_ms"] for r in routed]

    print(f"\n── Summary ─────────────────────────────────────────────────────────────")
    print(f"  Total sentences:  {n}  (fix={len(fix_cases)}, preserve={len(preserve_cases)})")
    print(f"  Router:           TP={tp}  TN={tn}  FP={fp}  FN={fn}  F1={f1:.2f}")
    print(f"  Corrector input:  {total} sentences ({len(routed_fix)} TP + {len(routed_preserve)} FP)")
    print(f"  Corrector fixed:  {corrector_changed_tp}/{len(routed_fix)} TP sentences changed")
    if routed_preserve:
        print(f"  Corrector damage: {corrector_changed_fp}/{len(routed_preserve)} FP sentences changed")

    # Corrector accuracy per fix category
    fix_cats = sorted({r["category"] for r in routed_fix})
    if fix_cats:
        print(f"\n  Corrector by fix category (among routed):")
        print(f"  {'Category':<22}  routed  changed  unchanged")
        for cat in fix_cats:
            group   = [r for r in routed_fix if r["category"] == cat]
            changed = sum(1 for r in group if r["changed"])
            print(f"  {cat:<22}  {len(group):6}  {changed:7}  {len(group)-changed}")

    print(f"\n  Latency:")
    print(f"    Router:    median={statistics.median(router_ms):.1f}ms")
    if corrector_ms:
        print(f"    Corrector: median={statistics.median(corrector_ms):.0f}ms  "
              f"p95={sorted(corrector_ms)[int(len(corrector_ms)*0.95)]:.0f}ms")
        print(f"    Clean pass (TN):   ~{statistics.median(router_ms):.1f}ms")
        print(f"    Routed (TP+FP):    ~{statistics.median(router_ms) + statistics.median(corrector_ms):.0f}ms")


if __name__ == "__main__":
    main()
