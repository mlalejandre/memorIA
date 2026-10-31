# LoCoMo manual audit (patch 16 run, 29/10/2026)

Run audited: the full LoCoMo run of patch 16 (correct 70.2 %, invents 5.8 %), compared against the original
turns of `locomo10.json` (sha256 79fa87e9…). Samples are deterministic (`random.Random(19)`).

## Single-hop failures with all evidence in the prompt (40 of 80)

| Cause | n | Example |
|---|---|---|
| LoCoMo label wrong or unanswerable from the text | 12 (30 %) | evidence spoken by the other person; question date ≠ turn date; answer only in the photo |
| Judge too strict (answer contains the reference) | 11 (28 %) | reference listed second, or with caveats |
| Genuine answer error | 10 (25 %) | picks the wrong candidate, refuses an obvious inference, loses the previous turn |
| Evidence turn not really in the prompt | 7 (18 %) | block-level coverage (6 turns) counted it as present |

## Temporal failures with all evidence in the prompt (30 of 61)

| Cause | n |
|---|---|
| Relative date resolved to the day it was said, not the day it happened | 10 (33 %) |
| Reasoning: refuses to compute a duration, picks another mention | 9 (30 %) |
| Label wrong (evidence says Tuesday, label Thursday…) | 7 (23 %) |
| Judge | 3 |
| Evidence not in prompt | 1 |

Over all 321 temporal questions: in 53 of the 77 failures whose evidence has a relative expression, the answer
gives the date of the message. Root cause: `extract_facts` drops "yesterday" and the source carries the date it
was said, which the agent takes as the event date → patch 21 (temporal anchoring at ingestion) tested it: not significant, reverted (see LEDGER.md).

## Inventions on adversarial questions (all 26)

| Cause | n |
|---|---|
| Label wrong (the asked person did say it) | 4 |
| Premise corrected, judged as invention | 2 |
| Notices the other speaker but still leads with the asked name | 5 |
| Pure misattribution | 15 |

Real invention ≈ 20 / 442 ≈ 4.5 % (wide interval).

## Consequences
- About a quarter of the audited failures are label errors: even a perfect system would stay around 93-95 %, so
  the 95 % objective is not reachable on LoCoMo as labelled. Raw scores are still reported unchanged.
- Failed answers are long (median 593 characters vs 332 for correct ones) and 46 failures contain the reference
  verbatim: answering the asked fact first, in one sentence, is a candidate for a later experiment.
