# Changelog

<!-- [PARCHE15] -->
## 0.3.0a3 (2026-10-29)
LoCoMo: correct 61.9 % → 70.2 % · multi-hop 35.5 % → 47.5 % · invents 7.0 % → 5.8 %. Own corpus unaffected (its tests use no episodes).

- LoCoMo evidence-depth diagnostic (`HMEM_LOCOMO_PROFUNDIDAD=1`, retrieval only with `HMEM_LOCOMO_SOLO_RECUPERACION=1`): rank of every evidence turn among facts and episodes, and simulated coverage for K facts + E episodes. Multi-hop full coverage: 19.1 % (current) → 29.4 % with 20 episodes.
<!-- [PARCHE16] -->
- `EPI_PROMPT_K` (default 5, unchanged): episodes shown in the prompt, previously hard-coded to 5 (so raising `EPI_LOOKUP_K` alone had no effect). The LoCoMo suite now measures the episodes really injected and reports the system-prompt size. Paired on LoCoMo, 20 episodes vs 5: correct 61.9 % → 70.2 % (175 better / 47 worse, p < 0.001), invention 7.0 % → 5.8 % (p = 0.46).
<!-- [PARCHE17] -->
- Defaults `EPI_PROMPT_K` = 20 and `EPI_LOOKUP_K` = 22. The LoCoMo suite reports the agent's answer time and only full runs are written as `resultados_*.jsonl` (comparison bases); partial runs go to `parcial_*.jsonl`. New frozen baseline `experiment/baseline_locomo_v0.3.0a3.json`. Answer time with 20 episodes vs 5: median 1.72 s vs 1.44 s, P95 2.38 s vs 2.46 s.
<!-- [PARCHE18] -->
- Fix: `EPI_LOOKUP_K` is raised to `EPI_PROMPT_K + 2` (with a warning) when it is lower. An old `.env` with `EPI_LOOKUP_K=6` silently limited the prompt to 6 episodes.

<!-- [PARCHE10] -->
## 0.3.0a2 (2026-10-29)
First measurement on a public benchmark. LoCoMo (10 conversations, 1,986 questions): correct 61.9 % · invents on adversarial questions 7.0 % · P95 115 ms. Own corpus unchanged (≈ 100 % / 0 %). The objective is NOT met on LoCoMo.

- New benchmark suite `locomo`: the public LoCoMo benchmark (Maharana et al., ACL 2024), ingested through the real pipeline (`extract_facts` → `store()` + episodic memory) and measured against the same objective. Its 446 adversarial questions (category 5) measure invention with far more cases than our 31 personal near-misses, and every failed answer is attributed to ingestion, retrieval, selection or answering.
- The dataset is downloaded at runtime and checked by sha256 (not redistributed). Ingested memories are cached in `experiment/locomo_cache/` (git-ignored) and ingestion is resumable.
<!-- [PARCHE11] -->
- LoCoMo diagnostics: identity of every injected fact (asked person / other person / anonymous "the user"), invention rate with and without anonymous facts in context, rank and coverage of the evidence (top-K), top-1 similarity of answerable vs adversarial questions, and the full context of each question in the audit file.
- No change to the memory's behaviour (LoCoMo suite and diagnostics).
<!-- [PARCHE12] -->
- Agent prompt: check that a remembered fact belongs to the person or thing asked about and correct the question's premise when it belongs to someone else; resolve relative time expressions from the memory's own date; in episodic memory, first-person statements belong to the speaker. Motivated by a manual audit of LoCoMo: 52 of 57 inventions attributed one person's statement to the other. Paired on LoCoMo: invention 12.8 % → 6.3 % (p < 0.001), correct answers unchanged; own corpus still 99.1 % / 100 % / 0 %.
<!-- [PARCHE13] -->
- `LTM_DIRECT_TOP_K`: number of facts injected in direct retrieval, previously hard-coded to 6.
<!-- [PARCHE14] -->
- Default `LTM_DIRECT_TOP_K` = 10. Paired on LoCoMo against 6: correct 59.7 % → 61.9 % (p = 0.003, 79 better / 45 worse), invention 6.3 % → 7.0 % (p = 0.65, not significant); own corpus 100 % / 100 % / 0 %.
- README: LoCoMo results next to our own, rewritten known limitations. New frozen baseline `experiment/baseline_v0.3.0a2.json` (own corpus at 108 and 100,108 facts + LoCoMo).

<!-- [PARCHE9] -->
## 0.3.0a1 (2026-10-28)
Objective, measured end to end with the real agent: correct answers ≥ 95 % · invented personal facts ≤ 5 % ·
retrieval P95 ≤ 1.5 s, and it must hold as memory grows.
**Result at 100,108 facts (×927 the base memory):** correct 99.1 % · memory-only (user) questions 100 % ·
invented 3.2 % (1/31) · P95 156 ms · first query after learning a fact 92 ms. Baseline: `experiment/baseline_v0.3.json`.

Changes, each one driven by a measurement:
- Uncertain ("tentative") memories now reach the prompt with an explicit caution (they were presented as facts).
- Real channel ablation with explicit weights: embeddings-only retrieval equals or beats the full RRF fusion and is
  2.7× faster. Channels with weight 0 (exact LIKE, FTS5, synapses) are no longer computed; they remain as fallback.
- New benchmark suites: `dificil` (dense same-domain corpus + near-miss questions), `reranker`, `escala`
  (10k / 100k realistic distractors), with an LLM judge that checks a verifiable property ("does the answer assert
  a value?") and a no-memory reference.
- Optional reranker `RERANK_MODE=llm|cross`. The LLM judge improves Hit@1 (82 → 90 %) but not the objective, at
  2–3× the latency, so the default is `off`.
- Agent prompt: personal facts are only stated if stored explicitly (no deductions).
- Vector search: incremental cache (adding a fact no longer re-reads every vector), covering index (kind, model),
  cache limit 20k → 200k vectors. With numpy, latency grows ≈ 0.5 ms per 1,000 facts.

## 0.2.0a1 (alpha)
- First public release.
- Hierarchical memory with adaptive-depth tree, atomic cells, sleep cycle (9 tasks), reversible forgetting.
- Hybrid retrieval (exact + FTS5 + embeddings + tree + synapses) fused with RRF.
- Multi-cartridge manager with import/export (path-traversal, size and schema checks).
- Benchmark (`benchmarks/prueba.py`) and unit tests.
