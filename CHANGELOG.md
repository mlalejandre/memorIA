# Changelog

<!-- [PARCHE19] -->
## Unreleased
<!-- [PARCHE32] -->
- The model selector knows which GPUs each `llama-server` build can run on: it reads every GPU's compute capability (`nvidia-smi`) and the CUDA architectures of each build (`build/CMakeCache.txt`). A GPU the build does not cover is shown disabled with the fix, instead of trying and rolling back. Found on the server: the build has `CMAKE_CUDA_ARCHITECTURES=86;89` (RTX 3090 and 4090), so the RTX 2060 (compute 7.5) cannot run it.
- `GPU_SERVER_BINS` (e.g. `2=/path/build-sm75/bin/llama-server`): a separate `llama-server` for one GPU, used only there. `scripts/modelos.py --compilar-gpu N` builds it on the server in `build-sm<arch>`, copying the existing build's options and changing only the CUDA architecture; `--latencia-emb` measures milliseconds per embedding call, idle and while the chat model on the same GPU is generating.
- Model names: a fragment that matches several files picks the one whose name ends with it (`GSQ-RCO-IQ3_S` is not the `-mtp` file).
- `parche.py --solo-eval` measures the embedding latency on each compatible GPU and, for the answering-model comparison, puts the long-term-memory server on the base's judge model while measuring and back afterwards, so no manual step is needed.
- Patch 31 result: the console panel switched models on the real server; moving the embeddings to the RTX 2060 failed and went back to the RTX 3090 (the reason above). The answering-model comparison was not run: `GSQ-RCO-IQ3_S` matched two files, and both servers had been set to Qwen3.8 in the console, which would also change the judge.
<!-- [PARCHE31] -->
- Choose the model, GPU and context of each server from the web console (panel "🧩 Modelos y GPUs") or from the terminal (`python3 scripts/modelos.py`). The panel lists the GPUs, which server runs on which GPU and the `.gguf` files on the server; a change restarts that server with its current command line, changing only `-m`, `-c` and the GPU, waits for `/health`, and goes back to the previous command line if the new model does not load. The agent is paused while a server changes. The choice is saved in `body/servidores.json` and overrides `.env`; choosing the `.env` values again removes it. The embedding model cannot be changed there (stored vectors depend on it), only its GPU.
- `REMOTE_MODEL_PATH_LTM` (long-term-memory model, defaults to `REMOTE_MODEL_PATH`) and `REMOTE_MODELS_DIR` (where to look for `.gguf` files; defaults to the parent folder of the known models).
- Patch 30 not measured yet: the answering-model comparison is carried over to this patch (`python3 parche.py --solo-eval --stm GSQ-RCO-IQ3_S`), which now switches the answering server itself and switches it back afterwards.
<!-- [PARCHE30] -->
- `REMOTE_MODEL_PATH_STM`: the answering server (GPU 1) can run a different model from the long-term-memory server, which keeps doing ingestion (and judging in the benchmark). Server launch commands set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so GPU indexes match `nvidia-smi`.
- The cartridge exam gains the mode `ref`: the same prompt answered by the long-term-memory server, to compare two answering models on exactly the same memory.
- Patch 29 result: `ATTRIB_EVENTS` rejected on the document exam (85.3 → 82.4 %: 0 better, 2 worse; traps 1/15 in both), so the LoCoMo phase was not needed; the console's one-source-per-document guard stays. Found on the server: the embedding server ran on the RTX 3090 next to the answering model (`CUDA_VISIBLE_DEVICES=1`) instead of the RTX 2060, which explains the 240-340 ms per embedding call since 2026-10-29.
<!-- [PARCHE29] -->
- Web console, document ingestion: the source field is cleared after each document, and the console asks for confirmation when a document has no source or reuses the previous document's source with different text. In patch 27 the second article inherited the first one's source and the agent mixed the two events.
- `ATTRIB_EVENTS` (default `0`, experimental): a prompt rule that extends attribution from people to events, dates and documents, and asks the agent to say a requested figure, name, cause or date is not recorded instead of completing it from general knowledge. The cartridge exam gains the mode `atrib` (same context, rule on).
- Patch 28 results (each text ingested with its own source into temporary memories): correct 85.3 % and 88.2 % in two identical ingestions (77.9 % in the patch 27 cartridge, whose facts all carried the first article's source: 7 better, 2 worse), traps invented 3/15 and 2/15. Self-contained facts: 86.8 %, 2/15 (3 better, 2 worse vs current; two identical ingestions already differ in 9 questions); not adopted, `DOC_SELF_CONTAINED` stays `0`.
<!-- [PARCHE28] -->
- Self-contained facts for document ingestion (`DOC_SELF_CONTAINED`, default `0`, `src/documentos.py`): each fact must name its event or subject and the absolute date when the text allows, with the document title and the end of the previous chunk as context. Only the console's document ingestion uses it; conversation ingestion (and LoCoMo) are unchanged.
- The cartridge exam can ingest texts itself into temporary memories (`--textos "path=source;..."`, variants `actual`, `actual2` (a second identical ingestion, to see how much ingestion alone varies) and `autocontenidos`), so no cartridge is touched. The two source texts are in `experiment/textos/` (CC BY-SA 4.0, attribution in `FUENTES.md`).
- Patch 27 result (cartridge `santander`, 83 questions): without memory 2.9 % correct; with memory 77.9 % correct and 1 of 15 traps invented; thinking did not help (Auto 79.4 %, Always 79.4 %, both 2-3 traps invented). Audit: every fact, including the 1941 fire, carried the source "Cabo Machichaco (vapor)"; several facts lost their event or date when extracted ("the cruiser Canarias arrived in port…"); and when a fact was missing the model sometimes filled the gap ("a badly stubbed cigarette") instead of saying it was not recorded.
<!-- [PARCHE27] -->
- Exam for a cartridge the user loaded (`benchmarks/cartucho.py`): works on a consistent copy of `body/cartuchos/<id>/memoria.db` (the original is only read), asks hand-written questions with reference answers and trap questions, and answers without memory and with memory under thinking off / auto / always, judged like LoCoMo. Evidence keys split failures into ingestion, retrieval and answer. First question set: `experiment/preguntas_santander.json` (83 questions on two Spanish Wikipedia articles, CC BY-SA 4.0: the Cabo Machichaco explosion of 1893 and the Santander fire of 1941, including traps that mix the two).
- Patch 26 results: neither experiment adopted. Auto thinking without durations: correct 69.6 → 70.1 % (26 better, 18 worse, p = 0.29), temporal 68.2 → 72.6 % (p = 0.016, mostly the same answers as patch 24). Fifteen facts in the prompt: 69.6 → 70.3 % (57/46, p = 0.32), invention 6.1 → 7.0 %.
<!-- [PARCHE26] -->
- `AGENT_THINK=auto` no longer thinks on duration and age questions ("how long…", "how long ago…", "¿hace cuánto…?"): in patch 24 thinking counted from the prompt's current date instead of repeating the duration stated in the memory, and these questions got worse.
- Web console: the "🧠 Pensar" selector gains "Por defecto" (uses `AGENT_THINK`), now the default choice, so a change of `AGENT_THINK` in `.env` reaches the console.
- Patch 25 (consolidation v0.3.0a4) accepted: full emulated suite with no new failures, LoCoMo reproduced exactly from cache (0 of 1,986 questions changed), own corpus at 100,108 facts 99.1 % correct / user 100 % / 1 of 31 invented / retrieval P95 446 ms. The embedding call alone took 244-339 ms median that night (it was ≈ 50-100 ms before), which explains the higher retrieval latency and the 83 minutes to vectorise 90,000 facts.

## 0.3.0a4 (2026-10-30)
<!-- [PARCHE25] -->
Consolidation of patches 19-24. Same defaults and answers as 0.3.0a3; the measurement is now reproducible and thinking is an option.
- Patch 24 results: one request at a time reproduces the base (69.6 % correct, 6.1 % invention; paired with the old stochastic measurement 35 better / 34 worse, p = 1.0) and a re-run without cache changed 0 of 99 verdicts. Automatic thinking (`AGENT_THINK=auto`) thought on 425 of 1,986 questions: correct 69.6 → 69.9 % (27 better, 22 worse, p = 0.57), temporal 68.2 → 72.3 % (p = 0.029), invention unchanged; 11.3 s vs 2.1 s per answer that thinks. Not adopted as default; `off` stays the default and `auto`/`on` are user options. Durations ("how long ago…") got worse when thinking: the model counted from the prompt's current date instead of the conversation date.
- Fixed: the security suite's stub agent did not accept the thinking mode forwarded by the console (`POST /api/chat` returned 400 in the benchmark only; the real agent was fine).
- Consolidation run: full emulated suite, LoCoMo reproduced from cache, own corpus re-measured at 100,108 facts, embedding-call latency measured separately. Frozen baselines `experiment/baseline_locomo_v0.3.0a4.json` and `experiment/baseline_v0.3.0a4.json`.
<!-- [PARCHE24] -->
- Optional thinking for the agent's final answer (`AGENT_THINK`: `off` default, `auto`, `on`; `src/pensar.py`). In `auto` the agent thinks only when the question asks it to compute something from memory (dates and durations, counts, combining facts about two people or moments) and skips direct and inference questions; every decision carries its reason. Ingestion, retrieval and judges are unchanged.
- Web console: a "🧠 Pensar" selector (No / Auto / Siempre) next to Send, remembered per browser, a "thinking" status while the model reasons and the reason in the answer metadata. `/api/chat` and `/api/chat/stream` accept `think`.
- Reproducible LoCoMo answers by default: one request at a time without server prompt cache (`HMEM_ANSWER_WORKERS=1`, `HMEM_ANSWER_CACHE_PROMPT=0`); the generation tag enters the cache key and the result fingerprint, so older answers are kept but not mixed in. `HMEM_ANSWER_THINK=auto` measures the automatic mode, with the paired comparison restricted to the questions that think.
- Patch 23 results: one request at a time repeats 100 % of answers at temperature 0 (two in parallel: 37-49 %) and answers in 1.80 s instead of 3.07 s, only 15 % lower throughput. Thinking on 397 questions: correct 69.6 → 73.1 % (22 better, 11 worse, p = 0.08; not adopted under the rule), invention 5.7 → 4.5 %, at 15.2 s instead of 2.96 s per answer.
<!-- [PARCHE23] -->
- Patch 22 result: `ANSWER_DIRECT` rejected. Correct 69.5 → 69.6 % (69 better, 68 worse, p = 1.0), invention 5.4 → 6.7 % (p = 0.35); answers were 28 % shorter and 7 % faster. Long failing answers were a symptom of missing knowledge, not the cause. The new benchmark cut verdict noise from 7.2 % to 2.7 %: retrieval and judge are reproducible; only the agent's wording varies at temperature 0 (35 % of answers identical across runs).
- Per-call `thinking` and `cache_prompt` in the LLM client, `LLM_CACHE_PROMPT` (default `1`, unchanged behaviour) and the token usage of the last answer.
- LoCoMo agent determinism diagnostic (`HMEM_LOCOMO_DETERMINISMO=1`): each question answered twice under 1 or 2 parallel requests, with and without server prompt cache; picks the fastest configuration that repeats ≥ 95 % of answers.
- LoCoMo thinking mode (`HMEM_ANSWER_THINK=1`, benchmark only): a probe first checks that the server thinks when asked; reports thinking tokens and answers cut by the limit (`HMEM_THINK_MAX_TOKENS`). Judge progress after the answers finish now counts only pending judgments.
<!-- [PARCHE22] -->
- Fast, reproducible LoCoMo benchmark: agent answers at temperature 0 and a single deterministic judge vote, both cached in `experiment/locomo_cache/banco.sqlite` under a key that covers the server fingerprint, the system prompt and the question, so unchanged questions are not recomputed and an interrupted run loses nothing. The judge runs on the idle long-term-memory server in parallel with the answers, with one progress line and an ETA. The prompt date is frozen (`HMEM_FIXED_NOW`). Each result gets a `.meta.json` with the server fingerprint (`/props`), and comparisons warn when the base was measured differently. Paired comparison per category.
- `ANSWER_DIRECT` (default `0`, experimental): the answer starts with one sentence that answers exactly what was asked.
- The web console always imports from the repository's `src/`, so modules added after `pip install -e .` no longer fail.
- `experiment/LEDGER.md`: scoreboard and history of every experiment and its decision. `experiment/audit_locomo_p16.md`: manual audit of the patch 16 run.
- Patch 21 (temporal anchoring at ingestion) was measured and rejected: temporal 68.8 → 70.4 % (p = 0.61), invention 5.8 → 8.1 % (p = 0.14). An A/A run showed 7.2 % of verdicts flip with nothing changed.
<!-- [PARCHE20] -->
- Episode diversity (`EPI_DIVERSITY=dia`, default `off`, experimental): episodes are searched deeper (`EPI_DIVERSITY_POOL`× `EPI_LOOKUP_K`) and at most `EPI_PER_DAY` per day reach the prompt, so turns from one session cannot fill the whole budget.
- Second retrieval pass (`LTM_SECOND_PASS`, default `0`, experimental): one LLM call reads what the first pass found and writes 1-3 concrete queries for what may be missing; results are fused with RRF under the same budget.
- LoCoMo diagnostic for both (`HMEM_LOCOMO_REFINAR=1`): evidence coverage for current retrieval, diversity, second pass, both and the oracle, plus the number of distinct sessions among prompt episodes, with a decision rule fixed before measuring.
- Decomposition diagnostic result (patch 19): NO-GO. LoCoMo multi-hop questions are mostly about one person with evidence across sessions, so the LLM decomposed only 9 % of them and full coverage did not move (30.5 → 30.5 %), while the oracle reached 59.2 %.
- Question decomposition (`LTM_DECOMPOSE`, default `0`, experimental): one LLM call rewrites a question that needs several facts into 2-3 single-fact search queries; facts and episodes retrieved for each are fused with RRF under the same budget (`LTM_DIRECT_TOP_K` facts, `EPI_PROMPT_K` episodes). Simple questions get no subqueries, so retrieval is unchanged.
- LoCoMo decomposition diagnostic (`HMEM_LOCOMO_DESCOMP=1`, retrieval only with `HMEM_LOCOMO_SOLO_RECUPERACION=1`): evidence coverage per category for current retrieval, decomposition and an oracle that searches with the reference answer, with a decision rule fixed before measuring. Subqueries are cached so the later experiment uses exactly the same ones.

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
