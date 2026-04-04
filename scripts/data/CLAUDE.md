# scripts/data/

## corrector_test_cases.json

Real transcription samples collected from production use. Each entry captures the full post-processing pipeline: raw ASR output through every stage (filler removal, ITN, vocab, neural grammar, cleanup) with timing and scoring metadata.

We collect these to build a regression suite for the grammar corrector. Cases demonstrate failure modes like the neural corrector inverting negations ("isn't working" to "is working") when the CoLA router scores compound nouns like "create snippet button" as low-acceptability.

When adding new cases, include the complete pipeline_stages array and chunk_timings so we can replay the full transformation chain.

## router_test_cases.json

Labeled examples for fine-tuning and evaluating the CoLA router (grammar acceptability classifier). Each entry is a sentence with a label: 1 = acceptable (should pass through), 0 = needs correction (should route to corrector).

When adding cases, focus on sentences the router currently gets wrong: clean text it flags as unacceptable (false positives) and broken text it lets through (false negatives).
