# Experiment ledger

Every patch either moves one of the four goals with a pre-registered decision rule, or is a declared diagnostic
that decides whether an experiment is worth running. Update this file in every patch.

## Scoreboard (current defaults)

| Goal | Metric | Now | Target |
|---|---|---|---|
| Accurate | LoCoMo correct, categories 1-4 (raw, LLM judge) | 69.6 % (reproducible base, P24; 70.2 % in P16 with the old judge) | ≥ 95 % raw is not reachable: ~¼ of audited failures are label errors |
| Accurate | Document exam (83 hand-written questions, two Wikipedia articles) | 85-88 % correct, 2/15 traps invented (P28-P32) | ≥ 95 % |
| Accurate | Own corpus correct at 100,108 facts | 99.1 % (user questions 100 %, v0.3.0a4) | ≥ 95 % |
| Reliable | LoCoMo invention on adversarial questions | 6.1 % (P24; ≈ 4.5 % after manual audit in P16) | ≤ 5 % |
| Reliable | Own corpus invention (31 near-miss questions) | 3.2 % | ≤ 5 % |
| Reliable | Benchmark noise: questions that flip with nothing changed | 0/99 (P24, one request at a time); 2.7 % in P22; 7.2 % before | as low as possible |
| Fast | Retrieval P95 (facts + episodes) | 115 ms | ≤ 1.5 s |
| Fast | Agent answer time, median / P95 (LLM included) | 1.80 s / 3.18 s one request at a time, as in real use (P23); 3.0 s / 6.0 s was a benchmark artefact of 2 parallel requests | — |
| Fast | Thinking mode, median per answer | 15.2 s with 2 in parallel (P23) | only where it pays off (`AGENT_THINK=auto`) |
| Fast | LoCoMo benchmark, full run | 53 min answers + judge overlapped; ≈ 5 min when nothing changed | — |
| Infinite | Retrieval P95 at 100,108 facts | 446 ms (v0.3.0a4; 357 ms in v0.3.0a2; the embedding call alone was 244-339 ms that night) | ≤ 1.5 s |
| Fast | One embedding call (short query) | ≈ 330 ms on any GPU, idle or loaded (P32); it should be a few ms | < 50 ms |

## History

| Patch | Type | Hypothesis | Result | Decision |
|---|---|---|---|---|
| P11 | diagnostic | anonymous "the user" facts cause invention (H1) | refuted; manual audit: 52/57 inventions attribute one person's words to the other | → P12 |
| P12 | experiment | attribution rule in the prompt | invention 12.8 → 6.3 % (p < 0.001), correct unchanged | adopted |
| P13 | experiment | inject 10 facts instead of 6 | correct 59.7 → 61.9 % (p = 0.003) | adopted |
| P15 | diagnostic | multi-hop evidence lies deep in the ranking | full coverage 19 → 29 % with 20 episodes | → P16 |
| P16 | experiment | 20 episodes in the prompt instead of 5 | correct 61.9 → 70.2 % (p < 0.001), multi-hop 35.5 → 47.5 %, invention 7.0 → 5.8 % | adopted |
| P18 | fix | old `.env` silently capped episodes | fixed | — |
| P19 | diagnostic | decomposing the question reaches multi-hop evidence | decomposed 9 % of multi-hop; full coverage 30.5 → 30.5 %; oracle 59.2 % | NO-GO |
| P20 | diagnostic | per-day episode diversity / second retrieval pass | +3.9 / +3.9 / +4.6 pp coverage (rule: ≥ 5 pp); prompt already holds 12 sessions | NO-GO |
| audit | diagnostic | where answers fail with the evidence in the prompt | label errors ~25-30 %, judge too strict ~20-28 %, temporal anchoring 33 % of temporal | → P21, P22 |
| P21 | experiment | temporal anchoring at ingestion | dated facts 5 → 17.6 %, temporal 68.8 → 70.4 % (p = 0.61), invention 5.8 → 8.1 % (p = 0.14) | rejected, reverted |
| A/A | measurement | how much moves with nothing changed | correct 70.2 → 69.7 %, invention 5.8 → 6.3 %, 111/1540 verdicts flip | → P22 |
| P22 | measurement | deterministic, cached benchmark (T=0, one judge vote on the idle server) | noise 7.2 → 2.7 %; base 69.5 % / invention 5.4 % (same as before, p = 0.83); 0 judge flips on identical answers; 35 % of agent answers identical across runs | adopted |
| P22 | experiment | direct answers (`ANSWER_DIRECT`) | correct 69.5 → 69.6 % (69/68, p = 1.0), invention 5.4 → 6.7 % (p = 0.35); answers −28 % length, −7 % time | rejected |
| P23 | diagnostic | agent determinism: 1 or 2 parallel requests × server prompt cache | 1 at a time: 100 % identical answers and 1.80 s vs 3.07 s; 2 in parallel: 37-49 % | adopted for the benchmark in P24 |
| P23 | experiment | thinking mode on 397 questions | correct 69.6 → 73.1 % (22/11, p = 0.08), invention 5.7 → 4.5 %, temporal +6.1 pp, inference −5 pp; 15.2 s vs 2.96 s | not adopted as default; offered as an option |
| P24 | measurement | reproducible base, one request at a time | 69.6 % / invention 6.1 % (vs 69.5 / 5.4 with 2 in parallel: 35/34, p = 1.0); re-run without cache 0/99 verdicts changed; answers 2.1 s | adopted |
| P24 | experiment | `AGENT_THINK=auto` (rules: dates, counts, comparisons) on full LoCoMo | 425/1,986 think; correct 69.6 → 69.9 % (27/22, p = 0.57); temporal +4.1 pp (22/9, p = 0.029, subgroup); single-hop 1/6; durations computed from the prompt date; 11.3 s vs 2.1 s | not default; user option (off/auto/on) |
| P25 | consolidation | v0.3.0a4: full emulated suite, LoCoMo reproduced from cache, own corpus at 100,108 facts, embedding latency | no new failures (and a benchmark stub fixed); LoCoMo 0/1,986 changed; own corpus 99.1 % / user 100 % / 1/31 / 446 ms; embedding call 244-339 ms median | accepted, tag v0.3.0a4 |
| P26 | experiment | `AGENT_THINK=auto` without durations (369 questions think) | correct 69.6 → 70.1 % (26/18, p = 0.29); temporal 68.2 → 72.6 % (22/8, p = 0.016; mostly the P24 answers, not independent) | not default; better user option than P24's auto |
| P26 | experiment | 15 facts in the prompt instead of 10 | correct 69.6 → 70.3 % (57/46, p = 0.32); invention 6.1 → 7.0 %; multi-hop −1.8, inference +4.1 pp | rejected |
| P27 | measurement | a document the user loaded (cartridge `santander`, two Wikipedia articles) with 83 hand-written questions | without memory 2.9 %; with memory 77.9 %, 1/15 traps; Auto 79.4 % (2/15), Always 79.4 % (3/15); 0 ingestion, 5 retrieval, 10 answer failures; all facts carried the first article's source | thinking not recommended for documents |
| P28 | experiment | self-contained facts at document ingestion, each article with its own source, vs current ingestion (and a second current ingestion as noise) | own sources: 85.3 / 88.2 % (two ingestions; 77.9 % with one shared source); self-contained 86.8 % (3/2); traps 2-3/15; cross-document questions 2-3/6 | rejected (source per document is what mattered) |
| P29 | experiment + fix | console asks for one source per document; `ATTRIB_EVENTS` rule (events, dates, documents; say not recorded) | document 85.3 → 82.4 % (0/2), traps 1/15 both; LoCoMo phase stopped (not needed) | rule rejected; console guard kept |
| P30 | experiment | answering model: Qwen3.8-27B (dense, GSQ-RCO IQ3_S) instead of nail-35b (MoE, 3B active), same memory and judge | not measured (servers not switched; carried over to P31) | — |
| P31 | feature (+ P30 experiment) | choose model, GPU and context per server from the console or `scripts/modelos.py`, with automatic rollback; `--solo-eval --stm` switches the answering server for the P30 comparison and back | panel works on the real server; embeddings on the RTX 2060 failed (build is sm_86/89 only) and rolled back; comparison not run (ambiguous model name; judge server also switched to Qwen3.8) | feature kept |
| P32 | experiment + feature | E: embedding latency per compatible GPU, idle and with the neighbouring chat model generating; GPU compatibility check, per-GPU `llama-server`; P30 comparison with the judge pinned to the base's model | embeddings ≈ 330 ms on all three GPUs, with or without load (not the GPU); Qwen3.8-27B-mtp answering: LoCoMo 68.9 → 68.6 % (16/17, p = 1.0), invention 6.8 → 3.4 % (4/1, p = 0.38), 8.4 s vs ≈ 1.8 s; document 86.8 → 85.3 %, traps 2/15 both, 5.0 s vs 1.2 s | embeddings stay; model not recommended; compatibility check and per-GPU build kept |
| P33 | consolidation | v0.3.0a5: markers removed, handoff (`resumen.txt`), patch tooling in `scripts/`, history archived outside the repo; tests + full emulated suite | see the patch summary | tag v0.3.0a5 |

## Lessons
- Effects of 1-3 pp were below the noise floor (7 % of verdicts flip): measure with a deterministic benchmark first.
- Cheap diagnostics (10-15 min, no answer LLM) closed three directions before spending ~2 h each on experiments.
- Restarting a server changed answer time by 75 %: record the server fingerprint with every result.
- How data is loaded can matter more than the algorithm: a shared source label cost about 8 points on the document exam (P27-P28).
- LoCoMo plateaued near 70 % with this model: the remaining knobs reshuffle which questions are right (K = 15 changed 103 answers, net +11). Next evidence must come from other material, starting with documents the user loads.
- A subgroup that passes (temporal, p = 0.029) is not the pre-registered rule; it is the hypothesis for the next experiment.
- Consolidate after a run of experiments: the full suite caught a benchmark stub broken by P24 that the LoCoMo-only runs missed.
- Correlation is not cause: failing answers were long (P16 audit), but forcing short answers changed which questions
  were right without changing how many (P22).
- A second answering model (dense 27B vs MoE 3B active) gave the same LoCoMo accuracy (P32): the bottleneck is what reaches the prompt and how, not the model. In 70 of 97 failures the evidence was in the prompt.
- Test the cheap explanation first: the embedding latency was blamed on the shared GPU for two days; measuring it on three GPUs took 3 minutes and refuted it (P32).
- Pin the judge: changing the long-term-memory model in the console also changes the benchmark judge; comparisons are only valid with the base's judge (P31-P32).
