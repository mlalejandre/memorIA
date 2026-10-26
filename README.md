# 🧠 Hierarchical Memory

> Hierarchical long-term memory for local LLMs: an adaptive-depth semantic tree (up to 7 levels) of nodes plus atomic "cells", an autonomous background maintenance loop, hybrid retrieval and per-fact provenance. **Python standard library only (`numpy` is optional).**
>
> ⚠️ **Status: alpha (v0.2.0a1).** It works and is tested, but it is a research project: see *Known limitations* below.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-0%20(stdlib)-brightgreen.svg)]()

The agent keeps **two memories queried in parallel** on every message:

| | Short-term (STM) | Long-term (LTM) |
|---|---|---|
| Holds | the current conversation | consolidated, sourced knowledge |
| Storage | `body/corto_plazo/sesion.json` | SQLite + FTS5 (`body/largo_plazo/memoria.db`) |
| Retrieval | local BM25 over old messages | exact + FTS5/BM25 + LLM query expansion, fused with RRF |
| Housekeeping | hierarchical auto-summarisation | "sleep cycle" (below) |

## ✨ What it actually does

- **Tree ontology.** Facts are filed in a tree of **adaptive depth** (1 to `MEM_MAX_DEPTH` levels, default 5; `task_split` can go down to 7), from general to specific concepts (the 7 conceptual levels `planeta ➜ país ➜ ciudad ➜ fábrica ➜ persona ➜ órgano ➜ tejido` are only a metaphor). Each fact is a **cell**: one atomic statement with source, importance and timestamps. Set `MEM_ADAPTIVE_DEPTH=0` for the legacy fixed 7-level layout.
- **Sleep cycle.** While idle, a loop runs nine tasks in rotation: `repair` (counters, empty-branch pruning, FTS5 rebuild), `embed` (missing vectors), `decay` (archive low-importance cells, prune stale synapses), `describe` (LLM-written node summaries), `dedupe`, `selftest` (asks itself a question about a cell and checks it can be recalled), `reevaluate`, `split` (rebalance overloaded leaves) and `merge`.
- **Forgetting that is reversible.** Archived cells are not deleted; they are restored automatically when a later query retrieves them.
- **Hybrid retrieval.** Exact match + SQLite FTS5 (BM25); when both find nothing, the LLM proposes related keywords. Rankings are fused with Reciprocal Rank Fusion and the LLM answers *only* from the retrieved cells.
- **Conversation → memory.** Facts in what the user says are extracted by the LLM into atomic statements and queued for storage in the background (`LTM_INGEST_USER`).
- **Provenance.** Every cell keeps its source (book/paper detected from the header, or the timestamped conversation) and answers cite it.
- **Local web console** at `http://127.0.0.1:8765` (UI in Spanish): chat, live pipeline, tree viewer, visual explorer/editor of the ontology, bulk ingestion of chapters/papers, diagnostics.

- **Tree navigation.** Retrieval also descends the tree with a beam search (`MEM_NAV`, `MEM_BEAM`). By default the branch choice is **vectorial/lexical, with no LLM call** (`MEM_NAV_LLM=0`); set `MEM_NAV_LLM=1` to let the LLM pick branches (slower, up to `MEM_NAV_MAX_CALLS` calls).
- **Episodic memory.** Every message is also stored verbatim in `memoria.db` (`episodes` + FTS5) and never forgotten or summarised; relevant literal snippets that are no longer in the live short-term memory are injected into the prompt (`EPI_ENABLED`, `EPI_LOOKUP_K`). Search it at `/api/episodes?q=...`.

- **Embeddings (optional, GPU 2).** A second `llama-server --embeddings` (bge-m3 by default, port `SERVER_PORT_GPU2`=8089) feeds a fifth RRF ranking for cells and episodes, nearest-neighbour placement in `store()` (0–4 LLM calls instead of 7), cheaper tree navigation and cosine-based dedupe. Vectors live in `memoria.db`; if the server is down everything works as before. `pip install numpy` (optional) speeds up the cosine search. Measure it with `python3 scripts/evaluar.py`.

## 🚀 Quick start

Requirements: Python 3.10+ (SQLite with FTS5, bundled with all mainstream builds) and any OpenAI-compatible endpoint (llama-server, vLLM, Ollama...).

```bash
git clone https://github.com/mlalejandre/memorIA.git
cd memorIA
cp .env.example .env        # point SERVER_IP / ports / REMOTE_MODEL_ALIAS at your server
python3 iniciar_interfaz.py # opens http://127.0.0.1:8765
```

With `SERVER_AUTOSTART=1` (and `REMOTE_MODEL_PATH`) the project can also launch `llama-server` for you, locally or over SSH. Servers it did not start are never stopped.

A terminal-only mode is available with `python3 src/main.py` (Ctrl+C pauses the loop so you can talk to the memory).

### Optional: install as a package

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # no sys.path tricks needed; adds `hmem-ui` and `hmem-cli`
hmem-ui
```

Without installing, running the scripts from the project root keeps working.

### Platform notes

- The web console, the memory engine and the tests run on Linux, macOS and Windows.
- **`SERVER_AUTOSTART=1` with a local `SERVER_IP` needs a POSIX shell (Linux, macOS or WSL2).** On native Windows use `SERVER_AUTOSTART=0` and start your LLM server yourself, or point `SERVER_IP` at a remote Linux host (launched over SSH).
- Chat answers stream token by token (Server-Sent Events); bulk ingestion reports progress block by block.
- To dump the project for sharing/review: `python3 scripts/estructura.py`.

### 🔌 Connecting to LLM Backends (Ollama, vLLM or llama-server)

The memory matrix connects to any OpenAI-compatible server. Configure `.env` according to your backend:

<details>
<summary><b>Option A: Ollama (Recommended for fast local testing)</b></summary>

```bash
ollama run llama3.1:8b
```

In your `.env`:
```ini
SERVER_IP=127.0.0.1
SERVER_PORT_GPU0=11434
SERVER_PORT_GPU1=11434
LLM_LTM_URL=http://127.0.0.1:11434/v1
LLM_STM_URL=http://127.0.0.1:11434/v1
REMOTE_MODEL_ALIAS=llama3.1:8b
SERVER_AUTOSTART=0
```
</details>

<details>
<summary><b>Option B: vLLM</b></summary>

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

In your `.env`:
```ini
SERVER_IP=127.0.0.1
SERVER_PORT_GPU0=8000
SERVER_PORT_GPU1=8000
LLM_LTM_URL=http://127.0.0.1:8000/v1
LLM_STM_URL=http://127.0.0.1:8000/v1
REMOTE_MODEL_ALIAS=Qwen/Qwen2.5-7B-Instruct
SERVER_AUTOSTART=0
```
</details>

<details>
<summary><b>Option C: llama-server (Built-in autostart)</b></summary>

Set `SERVER_AUTOSTART=1`, point `REMOTE_MODEL_PATH` to your `.gguf` file, and background instances will be managed automatically.
</details>

## ⚙️ Configuration

Everything is an environment variable or a line in `.env` (see [`.env.example`](.env.example)). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `SERVER_IP` | `127.0.0.1` | LLM host (anything else is reached/launched over SSH) |
| `SERVER_PORT_GPU0` / `_GPU1` | `8087` / `8088` | LTM server / STM+agent server (may be the same) |
| `SERVER_AUTOSTART` | `0` | launch `llama-server` if nothing is listening |
| `STM_MAX_TOKENS` | `30000` | short-term budget before auto-summarisation |
| `STM_ANSWER_MAX_TOKENS` | `4096` | max tokens per answer (a notice is shown when truncated) |
| `MEM_TAU` | `20` | forgetting time constant |
| `MEM_RECALL_EXPAND` | `1` | LLM keyword expansion when FTS finds nothing |
| `LTM_INGEST_USER` / `_ASSISTANT` | `1` / `0` | which messages are mined for facts |

## 📐 Forgetting curve

Vital facts persist far longer than trivia:

$$\text{effective importance} = I \cdot e^{-\frac{\text{age}}{\tau \cdot I^{3}}}$$

When it falls below `MEM_ARCHIVE_BELOW` the cell is archived (dormant). Age is measured in maintenance ticks, not wall-clock time.

## 🔒 Privacy & security

- Your conversations and memory live in `body/` (or `HMEM_HOME`) and are **git-ignored**. Never commit `memoria.db`, `sesion.json` or `.env`.
- The console only listens on `127.0.0.1` and rejects requests whose `Host`/`Origin` is not local (protection against CSRF and DNS rebinding from other web pages).
- Text coming from memory or the LLM is HTML-escaped before display.

## 🧪 Tests

```bash
python3 -m unittest discover -s tests -v
```

Benchmark with measurable goals (never touches your real memory; uses temporary folders):

```bash
python3 benchmarks/prueba.py          # emulated LLM/embeddings: validates logic, not semantic quality
python3 benchmarks/prueba.py --live   # real servers from .env
```

## 📄 License

MIT.
