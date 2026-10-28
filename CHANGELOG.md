# Changelog

<!-- [PARCHE10] -->
## Unreleased
- New benchmark suite `locomo`: the public LoCoMo benchmark (Maharana et al., ACL 2024), ingested through the real pipeline (`extract_facts` → `store()` + episodic memory) and measured against the same objective. Its 446 adversarial questions (category 5) measure invention with far more cases than our 31 personal near-misses, and every failed answer is attributed to ingestion, retrieval, selection or answering.
- The dataset is downloaded at runtime and checked by sha256 (not redistributed). Ingested memories are cached in `experiment/locomo_cache/` (git-ignored) and ingestion is resumable.
<!-- [PARCHE11] -->
- LoCoMo diagnostics: identity of every injected fact (asked person / other person / anonymous "the user"), invention rate with and without anonymous facts in context, rank and coverage of the evidence (top-K), top-1 similarity of answerable vs adversarial questions, and the full context of each question in the audit file.
- No change to the memory's behaviour (LoCoMo suite and diagnostics).
<!-- [PARCHE12] -->
- Agent prompt: check that a remembered fact belongs to the person or thing asked about and correct the question's premise when it belongs to someone else; resolve relative time expressions from the memory's own date; in episodic memory, first-person statements belong to the speaker. Motivated by a manual audit of LoCoMo: 52 of 57 inventions attributed one person's statement to the other.

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
