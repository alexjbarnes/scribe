#!/usr/bin/env python3
"""
General-public pipeline test: 270 cases.

Expands coverage beyond developer vocabulary to everyday language across
diverse domains: personal communication, shopping, family, healthcare, food,
travel, entertainment, and general office work.

Key test: proper nouns (names, brands, places) are the general-public analog
of technical abbreviations. This script tests whether the pipeline handles
them cleanly.

Categories:
  fix/homophone    - your/you're, their/there, by/buy, than/then, etc.
  fix/tense        - past narration errors in everyday speech
  fix/agreement    - subject-verb agreement in everyday context
  fix/informal     - "of" for "have" (should of, could of, etc.)
  fix/asr-dup      - ASR word repetitions
  fix/article      - missing or wrong a/an/the
  fix/plural       - wrong singular/plural form
  fix/contraction  - missing apostrophes (cant, doesnt, wont, etc.)
  preserve/everyday       - clean personal and domestic sentences
  preserve/work-general   - non-tech office and professional sentences
  preserve/proper-nouns   - names, brands, places (key FP test)
  preserve/habitual       - habitual present tense
  preserve/formal         - formal and polite register

Usage:
    python3 scripts/test_pipeline_v3.py
    python3 scripts/test_pipeline_v3.py --router-only
    python3 scripts/test_pipeline_v3.py --threshold 0.75
    python3 scripts/test_pipeline_v3.py --runs 2
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
    ("your going to love this restaurant",                         "fix/homophone"),
    ("their coming over for dinner on Sunday",                     "fix/homophone"),
    ("its going to be a great weekend",                            "fix/homophone"),
    ("make sure your ready before they arrive",                    "fix/homophone"),
    ("I think there going to be late",                             "fix/homophone"),
    ("we need to look at there schedule",                          "fix/homophone"),
    ("its the best movie I have seen this year",                   "fix/homophone"),
    ("your going to need a bigger bag",                            "fix/homophone"),
    ("she said there ready to leave",                              "fix/homophone"),
    ("the food is to hot to eat right now",                        "fix/homophone"),
    ("we have to many leftovers in the fridge",                    "fix/homophone"),
    ("the store is to far to walk from here",                      "fix/homophone"),
    ("the music is to loud in here",                               "fix/homophone"),
    ("I need to by groceries on the way home",                     "fix/homophone"),
    ("she past the driving test on her first try",                 "fix/homophone"),
    ("I want to no what time the movie starts",                    "fix/homophone"),
    ("the weather is nicer then I expected",                       "fix/homophone"),
    ("this dress is more expensive then the other one",            "fix/homophone"),
    ("he runs faster then anyone on the team",                     "fix/homophone"),
    ("I left my keys write next to the door",                      "fix/homophone"),
    ("we meat at the coffee shop every Tuesday",                   "fix/homophone"),
    ("I need to loose a few pounds before the wedding",            "fix/homophone"),
    ("I herd the news this morning",                               "fix/homophone"),
    ("the store is having a sell this weekend",                    "fix/homophone"),
    ("we went to there house for the holidays",                    "fix/homophone"),
    ("she through the ball really hard",                           "fix/homophone"),
    ("he one the championship last year",                          "fix/homophone"),
    ("it has been a hole week since I heard from her",             "fix/homophone"),
    ("I need a brake from all this work",                          "fix/homophone"),
    ("I heard the price was way to high",                          "fix/homophone"),

    # ── fix/tense ─────────────────────────────────────────────────────────────
    ("yesterday I call my mom and she sound really tired",         "fix/tense"),
    ("last night I cook a big pasta dinner for everyone",          "fix/tense"),
    ("I wake up late this morning and miss the bus",               "fix/tense"),
    ("she walk into the room and everyone stop talking",           "fix/tense"),
    ("we drive to the beach last Saturday",                        "fix/tense"),
    ("he forget to lock the door when he leave",                   "fix/tense"),
    ("I drop my phone in the sink this morning",                   "fix/tense"),
    ("they arrive at the airport three hours early",               "fix/tense"),
    ("I spend the whole afternoon cleaning the house",             "fix/tense"),
    ("she tell me about the party last night",                     "fix/tense"),
    ("we watch three movies this weekend",                         "fix/tense"),
    ("he ask me to help him move last weekend",                    "fix/tense"),
    ("I bake a cake for her birthday yesterday",                   "fix/tense"),
    ("they cancel the reservation without telling us",             "fix/tense"),
    ("I lose my wallet on the way to work yesterday",              "fix/tense"),
    ("she start a new job at the hospital last month",             "fix/tense"),
    ("we paint the living room last weekend",                      "fix/tense"),
    ("he graduate from college last spring",                       "fix/tense"),
    ("I sign up for the gym this morning",                         "fix/tense"),
    ("the flight land early and we have extra time",               "fix/tense"),
    ("they move to a new apartment last month",                    "fix/tense"),
    ("I receive a letter from the bank this week",                 "fix/tense"),
    ("we find a great parking spot right away",                    "fix/tense"),
    ("he break his arm playing football last weekend",             "fix/tense"),
    ("I order the wrong item by mistake yesterday",                "fix/tense"),
    ("the package arrive two days late",                           "fix/tense"),
    ("she win the raffle at the school fundraiser",                "fix/tense"),
    ("we stop at a gas station on the way there",                  "fix/tense"),
    ("I notice a scratch on the car this morning",                 "fix/tense"),
    ("she propose to him at the restaurant last night",            "fix/tense"),

    # ── fix/agreement ─────────────────────────────────────────────────────────
    ("my sister don't like spicy food",                            "fix/agreement"),
    ("the kids was really excited about the trip",                 "fix/agreement"),
    ("he don't want to go to the party",                           "fix/agreement"),
    ("the store don't open until nine",                            "fix/agreement"),
    ("she don't know about the surprise yet",                      "fix/agreement"),
    ("the movie were really good",                                 "fix/agreement"),
    ("my friends was late to the restaurant",                      "fix/agreement"),
    ("the traffic were terrible this morning",                     "fix/agreement"),
    ("he walk his dog every evening",                              "fix/agreement"),
    ("she take the train to work every day",                       "fix/agreement"),
    ("my dad fix cars in the garage on weekends",                  "fix/agreement"),
    ("the cat sit in the same spot every morning",                 "fix/agreement"),
    ("the price were higher than expected",                        "fix/agreement"),
    ("everyone in the family love the new house",                  "fix/agreement"),
    ("the team were playing really well until halftime",           "fix/agreement"),
    ("the restaurant close early on Sundays",                      "fix/agreement"),
    ("my neighbor play loud music every night",                    "fix/agreement"),
    ("the store have a big sale this weekend",                     "fix/agreement"),
    ("she teach piano to kids in the neighborhood",                "fix/agreement"),
    ("the bus arrive late every morning",                          "fix/agreement"),

    # ── fix/informal ──────────────────────────────────────────────────────────
    ("I should of brought an umbrella",                            "fix/informal"),
    ("we could of left earlier to avoid traffic",                  "fix/informal"),
    ("she would of loved this restaurant",                         "fix/informal"),
    ("they should of called ahead for a reservation",              "fix/informal"),
    ("I could of sworn I left my keys here",                       "fix/informal"),
    ("we might of missed the best part of the movie",              "fix/informal"),
    ("he should of studied harder for the exam",                   "fix/informal"),
    ("I would of helped if I had known",                           "fix/informal"),
    ("she must of forgotten about the appointment",                "fix/informal"),
    ("we could of gotten a better deal elsewhere",                 "fix/informal"),
    ("they must of taken the wrong exit",                          "fix/informal"),
    ("I should of asked for directions earlier",                   "fix/informal"),
    ("you would of laughed if you were there",                     "fix/informal"),
    ("we could of avoided the whole argument",                     "fix/informal"),
    ("she might of already left by now",                           "fix/informal"),

    # ── fix/asr-dup ───────────────────────────────────────────────────────────
    ("so I was saying we could could go out for dinner",           "fix/asr-dup"),
    ("I I just wanted to let you know I will be late",             "fix/asr-dup"),
    ("can you you pick up some milk on the way home",              "fix/asr-dup"),
    ("the the concert was amazing last night",                     "fix/asr-dup"),
    ("I need to to make a doctor's appointment",                   "fix/asr-dup"),
    ("we we should go on a vacation this summer",                  "fix/asr-dup"),
    ("let me me check the calendar for that date",                 "fix/asr-dup"),
    ("the weather the weather has been really nice lately",        "fix/asr-dup"),
    ("she said she she would call me back this evening",           "fix/asr-dup"),
    ("I want to to reserve a table for two",                       "fix/asr-dup"),
    ("can we we talk about the plans for this weekend",            "fix/asr-dup"),
    ("the the kids are already in bed",                            "fix/asr-dup"),
    ("I I really enjoyed the dinner last night",                   "fix/asr-dup"),
    ("we need need to leave in about ten minutes",                 "fix/asr-dup"),
    ("please please don't forget to water the plants",             "fix/asr-dup"),

    # ── fix/article ───────────────────────────────────────────────────────────
    ("I need to make appointment with the dentist",                "fix/article"),
    ("we had wonderful time at the wedding",                       "fix/article"),
    ("can you give me ride to the station",                        "fix/article"),
    ("she got job offer from the restaurant downtown",             "fix/article"),
    ("I am looking for apartment in this neighborhood",            "fix/article"),
    ("we saw incredible sunset from the beach",                    "fix/article"),
    ("I need to send email to my landlord",                        "fix/article"),
    ("he bought brand new car last week",                          "fix/article"),
    ("can you make reservation for tomorrow night",                "fix/article"),
    ("I have important meeting this afternoon",                    "fix/article"),
    ("she is nurse at the local hospital",                         "fix/article"),
    ("we need to find plumber for the weekend",                    "fix/article"),
    ("I found great recipe for chicken soup",                      "fix/article"),
    ("he gave me helpful tip for the trip",                        "fix/article"),
    ("I need to book flight to visit my family",                   "fix/article"),
    ("she is wearing a amazing dress",                             "fix/article"),
    ("I took a Uber to the airport",                               "fix/article"),
    ("she has an useful skill",                                    "fix/article"),
    ("it was a honor to meet her",                                 "fix/article"),
    ("I got a offer on the house",                                 "fix/article"),

    # ── fix/plural ────────────────────────────────────────────────────────────
    ("I bought three new shirt for the trip",                      "fix/plural"),
    ("she has two dog and one cat",                                "fix/plural"),
    ("we need four more chair for the table",                      "fix/plural"),
    ("the store sells different type of cheese",                   "fix/plural"),
    ("he has been to many different country",                      "fix/plural"),
    ("she wrote four thank you card",                              "fix/plural"),
    ("we planted six rose bush in the garden",                     "fix/plural"),
    ("there are three doctor in this practice",                    "fix/plural"),
    ("I ordered two coffee from the cafe",                         "fix/plural"),
    ("she submitted three job application this week",              "fix/plural"),

    # ── fix/contraction ───────────────────────────────────────────────────────
    ("I cant believe how fast the year went by",                   "fix/contraction"),
    ("she doesnt want to come to the party",                       "fix/contraction"),
    ("we wont be home until late tonight",                         "fix/contraction"),
    ("he didnt tell me about the change of plans",                 "fix/contraction"),
    ("I wasnt expecting that kind of response",                    "fix/contraction"),
    ("they arent coming to the dinner tonight",                    "fix/contraction"),
    ("it doesnt matter what time we leave",                        "fix/contraction"),
    ("I havent been to that restaurant yet",                       "fix/contraction"),
    ("she wouldnt take no for an answer",                          "fix/contraction"),
    ("we shouldnt have waited so long",                            "fix/contraction"),
    ("he hasnt called me back yet",                                "fix/contraction"),
    ("I couldnt find parking near the venue",                      "fix/contraction"),
    ("they dont know what they are missing",                       "fix/contraction"),
    ("I wouldnt worry about it too much",                          "fix/contraction"),
    ("she isnt feeling well today",                                "fix/contraction"),

    # ── preserve/everyday ─────────────────────────────────────────────────────
    ("I need to pick up the kids from school at three.",           "preserve/everyday"),
    ("We are having dinner at my parents' house tonight.",         "preserve/everyday"),
    ("Can you remind me to call the dentist tomorrow?",            "preserve/everyday"),
    ("I am going to the gym after work today.",                    "preserve/everyday"),
    ("The groceries are in the bag on the counter.",               "preserve/everyday"),
    ("We should watch a movie tonight.",                           "preserve/everyday"),
    ("I am really tired from all the cooking today.",              "preserve/everyday"),
    ("She called to say she will be there by seven.",              "preserve/everyday"),
    ("I need to send a birthday card to my aunt.",                 "preserve/everyday"),
    ("We are planning a trip to visit my sister.",                 "preserve/everyday"),
    ("Can you turn down the music a little?",                      "preserve/everyday"),
    ("I forgot to put the clothes in the dryer.",                  "preserve/everyday"),
    ("The dog needs to go for a walk before dinner.",              "preserve/everyday"),
    ("I think I left my phone in the car.",                        "preserve/everyday"),
    ("We should book the hotel room early this year.",             "preserve/everyday"),
    ("The kids have a school play this Friday evening.",           "preserve/everyday"),
    ("I am almost out of coffee.",                                 "preserve/everyday"),
    ("She got a new haircut and it looks great.",                  "preserve/everyday"),
    ("We need to call the plumber about the kitchen sink.",        "preserve/everyday"),
    ("I am going to bed early tonight.",                           "preserve/everyday"),
    ("The neighbors are having a party this weekend.",             "preserve/everyday"),
    ("I need to renew my driver's license next month.",            "preserve/everyday"),
    ("She moved to a new apartment closer to her job.",            "preserve/everyday"),
    ("We had a great time at the concert last night.",             "preserve/everyday"),
    ("I am taking the day off tomorrow to run some errands.",      "preserve/everyday"),
    ("Can you grab some bread and eggs from the store?",           "preserve/everyday"),
    ("The flight is at seven in the morning.",                     "preserve/everyday"),
    ("We are thinking about getting a new couch.",                 "preserve/everyday"),
    ("I have a dentist appointment on Wednesday morning.",         "preserve/everyday"),
    ("My sister is coming to stay with us for the holidays.",      "preserve/everyday"),
    ("I have been trying to get more sleep lately.",               "preserve/everyday"),
    ("She recommended a great book to me yesterday.",              "preserve/everyday"),
    ("We need to decide on a restaurant for Saturday night.",      "preserve/everyday"),
    ("I signed up for a cooking class that starts next week.",     "preserve/everyday"),
    ("The kids are doing really well in school this year.",        "preserve/everyday"),

    # ── preserve/work-general ─────────────────────────────────────────────────
    ("I need to prepare for the presentation on Friday.",          "preserve/work-general"),
    ("Can you send me the spreadsheet by end of day?",            "preserve/work-general"),
    ("The client meeting has been moved to Thursday.",            "preserve/work-general"),
    ("I am working on the quarterly report this week.",           "preserve/work-general"),
    ("She manages the accounts for three major clients.",         "preserve/work-general"),
    ("The conference call is scheduled for two o'clock.",         "preserve/work-general"),
    ("I need to follow up with the supplier about the delay.",    "preserve/work-general"),
    ("We are opening a new office location next quarter.",        "preserve/work-general"),
    ("The performance reviews are due at the end of the month.",  "preserve/work-general"),
    ("I sent the invoice to the client this morning.",            "preserve/work-general"),
    ("She was promoted to district manager last month.",          "preserve/work-general"),
    ("We need more staff during the holiday season.",             "preserve/work-general"),
    ("I have back-to-back meetings all morning.",                 "preserve/work-general"),
    ("Can you cover my shift on Saturday?",                       "preserve/work-general"),
    ("I am training a new employee this week.",                   "preserve/work-general"),
    ("The store closes at nine on weekdays.",                     "preserve/work-general"),
    ("We need to reorder supplies before we run out.",            "preserve/work-general"),
    ("She handles customer complaints for the whole team.",       "preserve/work-general"),
    ("The warehouse ships orders every Tuesday and Thursday.",    "preserve/work-general"),
    ("I sent a follow-up email after the interview.",             "preserve/work-general"),

    # ── preserve/proper-nouns ─────────────────────────────────────────────────
    # Tests that names, brands, and places do not trigger false positives.
    ("I need to pick up my order from Amazon.",                   "preserve/proper-nouns"),
    ("We are going to Disneyland in June.",                       "preserve/proper-nouns"),
    ("I ran into Sarah at the grocery store yesterday.",          "preserve/proper-nouns"),
    ("We are flying to Miami for the long weekend.",              "preserve/proper-nouns"),
    ("I bought this at Target last week.",                        "preserve/proper-nouns"),
    ("My friend Michael got married in Las Vegas.",               "preserve/proper-nouns"),
    ("I ordered a burrito from Chipotle for lunch.",              "preserve/proper-nouns"),
    ("We are moving to Austin next spring.",                      "preserve/proper-nouns"),
    ("Can you get me an Advil from the cabinet?",                 "preserve/proper-nouns"),
    ("I watched the new episode of The Crown last night.",        "preserve/proper-nouns"),
    ("The Lakers won by twelve points last night.",               "preserve/proper-nouns"),
    ("I got a gift card to Starbucks for my birthday.",           "preserve/proper-nouns"),
    ("We spent the weekend hiking in Yosemite.",                  "preserve/proper-nouns"),
    ("She is studying at NYU this fall.",                         "preserve/proper-nouns"),
    ("I called my mom on FaceTime last night.",                   "preserve/proper-nouns"),
    ("We went to Olive Garden for my dad's birthday.",            "preserve/proper-nouns"),
    ("I use Spotify for music and Netflix for shows.",            "preserve/proper-nouns"),
    ("The kids are obsessed with Minecraft right now.",           "preserve/proper-nouns"),
    ("He drives a Toyota Camry.",                                 "preserve/proper-nouns"),
    ("I renewed my Costco membership last month.",                "preserve/proper-nouns"),
    ("She got a job offer from Goldman Sachs.",                   "preserve/proper-nouns"),
    ("We stayed at the Marriott near the convention center.",     "preserve/proper-nouns"),
    ("My doctor is at the Cleveland Clinic.",                     "preserve/proper-nouns"),
    ("We are planning a road trip along Route 66.",               "preserve/proper-nouns"),
    ("I put the appointment in Google Calendar for next Friday.", "preserve/proper-nouns"),

    # ── preserve/habitual ─────────────────────────────────────────────────────
    ("I walk the dog every morning before breakfast.",            "preserve/habitual"),
    ("She calls her mom every Sunday afternoon.",                 "preserve/habitual"),
    ("We eat dinner as a family every night.",                    "preserve/habitual"),
    ("He goes to the gym three times a week.",                    "preserve/habitual"),
    ("I make coffee before I do anything else in the morning.",   "preserve/habitual"),
    ("They visit her parents every other weekend.",               "preserve/habitual"),
    ("The kids take the school bus every morning.",               "preserve/habitual"),
    ("I do yoga every Tuesday and Thursday.",                     "preserve/habitual"),
    ("She reads a chapter before bed every night.",               "preserve/habitual"),
    ("We order pizza every Friday night.",                        "preserve/habitual"),
    ("The mail arrives around noon every day.",                   "preserve/habitual"),
    ("I check my email first thing every morning.",               "preserve/habitual"),
    ("He volunteers at the food bank every Saturday.",            "preserve/habitual"),
    ("We watch the game together every Sunday.",                  "preserve/habitual"),
    ("The store offers a senior discount every Tuesday.",         "preserve/habitual"),
    ("I take a walk after lunch every day.",                      "preserve/habitual"),
    ("She tutors students every weekday after school.",           "preserve/habitual"),
    ("We have family movie night every Friday.",                  "preserve/habitual"),
    ("I batch cook meals every Sunday for the week.",             "preserve/habitual"),
    ("The library hosts storytime every Wednesday morning.",      "preserve/habitual"),

    # ── preserve/formal ───────────────────────────────────────────────────────
    ("We appreciate your patience during the transition.",        "preserve/formal"),
    ("Please confirm your attendance by Friday.",                 "preserve/formal"),
    ("The event will be held at the downtown convention center.", "preserve/formal"),
    ("Your application has been received and is under review.",   "preserve/formal"),
    ("We regret to inform you that the position has been filled.", "preserve/formal"),
    ("Please bring a valid form of identification to the appointment.", "preserve/formal"),
    ("The terms of the agreement are outlined in the attached document.", "preserve/formal"),
    ("We will be in touch once a decision has been made.",        "preserve/formal"),
    ("Your feedback is important to us.",                         "preserve/formal"),
    ("The event has been postponed to a later date.",             "preserve/formal"),
    ("We thank you for your continued support.",                  "preserve/formal"),
    ("Please review the attached document before the meeting.",   "preserve/formal"),
    ("The proposal has been submitted for final approval.",       "preserve/formal"),
    ("Guests are encouraged to arrive fifteen minutes early.",    "preserve/formal"),
    ("The program runs from nine in the morning until five in the afternoon.", "preserve/formal"),
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

    cats = sorted({r["category"] for r in router_results})
    print(f"\n  Per-category routing:")
    print(f"  {'Category':<26}  total  routed  FP/FN")
    for cat in cats:
        group    = [r for r in router_results if r["category"] == cat]
        routed_g = sum(1 for r in group if r["route"])
        if cat.startswith("fix"):
            miss = sum(1 for r in group if not r["route"])
            print(f"  {cat:<26}  {len(group):5}  {routed_g:6}  FN={miss}")
        else:
            false_pos = sum(1 for r in group if r["route"])
            print(f"  {cat:<26}  {len(group):5}  {routed_g:6}  FP={false_pos}")

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

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"\nLoading corrector: {CORRECTOR_MODEL} ...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(CORRECTOR_MODEL)
    mod = AutoModelForSeq2SeqLM.from_pretrained(CORRECTOR_MODEL)
    corrector = hf_pipeline("text2text-generation", model=mod, tokenizer=tok)
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

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

    col = 52
    if routed_fix:
        print(f"\n── Corrector on TP cases ────────────────────────────────────────────────")
        for r in routed_fix:
            marker = "~" if r["changed"] else "="
            inp = r["text"][:col] + "…" if len(r["text"]) > col else r["text"]
            out = r["corrected"][:col] + "…" if len(r["corrected"]) > col else r["corrected"]
            print(f"  {marker} {inp:<{col}}  →  {out}  ({r['corrector_ms']:.0f}ms)  [{r['category']}]")

    if routed_preserve:
        print(f"\n── Corrector on FP cases (should not have been routed) ─────────────────")
        for r in routed_preserve:
            marker = "!" if r["changed"] else "="
            inp = r["text"][:col] + "…" if len(r["text"]) > col else r["text"]
            out = r["corrected"][:col] + "…" if len(r["corrected"]) > col else r["corrected"]
            print(f"  {marker} {inp:<{col}}  →  {out}  ({r['corrector_ms']:.0f}ms)  [{r['category']}]")

    corrector_changed_tp  = sum(1 for r in routed_fix      if r["changed"])
    corrector_changed_fp  = sum(1 for r in routed_preserve if r["changed"])
    corrector_ms          = [r["corrector_ms"] for r in routed]

    print(f"\n── Summary ─────────────────────────────────────────────────────────────")
    print(f"  Total:            {n}  (fix={len(fix_cases)}, preserve={len(preserve_cases)})")
    print(f"  Router:           TP={tp}  TN={tn}  FP={fp}  FN={fn}  F1={f1:.2f}")
    print(f"  Corrector input:  {total}  ({len(routed_fix)} TP + {len(routed_preserve)} FP)")
    print(f"  Corrector fixed:  {corrector_changed_tp}/{len(routed_fix)} TP sentences changed")
    if routed_preserve:
        print(f"  Corrector damage: {corrector_changed_fp}/{len(routed_preserve)} FP sentences changed")

    fix_cats = sorted({r["category"] for r in routed_fix})
    if fix_cats:
        print(f"\n  Corrector by fix category (routed only):")
        print(f"  {'Category':<26}  routed  changed  unchanged")
        for cat in fix_cats:
            group   = [r for r in routed_fix if r["category"] == cat]
            changed = sum(1 for r in group if r["changed"])
            print(f"  {cat:<26}  {len(group):6}  {changed:7}  {len(group)-changed}")

    print(f"\n  Latency:")
    print(f"    Router:    median={statistics.median(router_ms):.1f}ms")
    if corrector_ms:
        print(f"    Corrector: median={statistics.median(corrector_ms):.0f}ms  "
              f"p95={sorted(corrector_ms)[int(len(corrector_ms)*0.95)]:.0f}ms")
        print(f"    Clean pass (TN): ~{statistics.median(router_ms):.1f}ms")
        print(f"    Routed path:     ~{statistics.median(router_ms) + statistics.median(corrector_ms):.0f}ms")


if __name__ == "__main__":
    main()
