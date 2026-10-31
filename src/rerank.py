# -*- coding: utf-8 -*-
"""rerank.py — Reordenación de candidatos y decisión de relevancia.

Modos (RERANK_MODE):
  * "llm":   juez LLM. El modelo de la memoria larga indica qué candidatos RESPONDEN de verdad a la pregunta
             (mismo sujeto y mismo atributo). 1 llamada por consulta.
  * "cross": cross-encoder (bge-reranker-v2-m3) servido por llama-server --reranking en RERANK_URL (/v1/rerank).
  * "off":   sin reranker (compuerta clásica: embeddings + solapamiento léxico).
Si el reranker falla o devuelve algo inválido, recall() vuelve a la compuerta clásica.
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from typing import Any

import config as C

JUDGE_SYSTEM = ("Eres el evaluador de relevancia de una memoria a largo plazo. "
                "Responde SIEMPRE y SOLO con un JSON válido, sin texto adicional.")

# Los ejemplos de la regla 2 son deliberadamente DISTINTOS de las preguntas del benchmark
# para no contaminar la medición.
JUDGE_PROMPT = (
    "Pregunta: {q}\n\n"
    "Recuerdos candidatos:\n{docs}\n\n"
    "Indica qué recuerdos RESPONDEN DIRECTAMENTE a la pregunta, ordenados del más útil al menos útil.\n"
    "Reglas:\n"
    "1. Un recuerdo responde solo si trata del MISMO sujeto y del MISMO atributo por el que se pregunta.\n"
    "2. No responde un recuerdo sobre otra entidad parecida (otra persona, otro objeto, otro elemento de una "
    "serie, otra fecha): por ejemplo, un dato del jefe no responde por el compañero, ni uno del móvil por el "
    "portátil, ni uno del año pasado por este año.\n"
    "3. Una pregunta indirecta SÍ se responde con un recuerdo que contenga la causa, el mecanismo o el nombre "
    "pedido, aunque use otras palabras.\n"
    "4. Si ninguno responde, devuelve una lista vacía. No adivines.\n"
    'Devuelve SOLO este JSON: {{"responden": [números de los recuerdos]}}'
)


def mode() -> str:
    return str(getattr(C, "RERANK_MODE", "off") or "off").strip().lower()


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def parse_rerank_response(data: Any, n: int) -> list[float]:
    """Respuesta de /v1/rerank → una puntuación por documento, en el orden ORIGINAL de los documentos."""
    items = (data.get("results") or data.get("data")) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("respuesta de rerank sin 'results'")
    scores: list[float | None] = [None] * n
    for it in items:
        idx = int(it.get("index"))
        s = it.get("relevance_score", it.get("score"))
        if 0 <= idx < n and s is not None:
            scores[idx] = float(s)
    if any(s is None for s in scores):
        raise ValueError("respuesta de rerank incompleta")
    if str(getattr(C, "RERANK_SIGMOID", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return [float(s) for s in scores]
    return [_sigmoid(float(s)) for s in scores]   # llama-server devuelve logits


class Reranker:
    def __init__(self, tree: Any) -> None:
        self.tree = tree
        self.calls = 0
        self.failures = 0
        self._fail_until = 0.0

    def enabled(self) -> bool:
        return mode() in ("llm", "cross") and time.time() >= self._fail_until

    def _cross_scores(self, question: str, docs: list[str]) -> list[float]:
        url = str(getattr(C, "RERANK_URL", "")).rstrip("/") + "/rerank"
        payload = {"model": getattr(C, "RERANK_MODEL", "bge-reranker-v2-m3"), "query": question,
                   "documents": [str(d)[:2000] for d in docs], "top_n": len(docs)}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=float(getattr(C, "RERANK_TIMEOUT_S", 60))) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return parse_rerank_response(data, len(docs))

    def _judge(self, question: str, docs: list[str]) -> list[int] | None:
        listing = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
        obj = self.tree.llm.ask_json(JUDGE_SYSTEM, JUDGE_PROMPT.format(q=question, docs=listing))
        if not isinstance(obj, dict) or not isinstance(obj.get("responden"), list):
            return None
        picks: list[int] = []
        for p in obj["responden"]:
            try:
                i = int(str(p).strip("[]() ")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(docs) and i not in picks:
                picks.append(i)
        return picks

    def rerank(self, question: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Reordena los primeros RERANK_TOP_K candidatos y decide si alguno responde (verdict)."""
        m = mode()
        k = max(2, int(getattr(C, "RERANK_TOP_K", 8)))
        head, tail = list(candidates[:k]), list(candidates[k:])
        if not head:
            return None
        docs = [str(c.get("text", "")) for c in head]
        t0 = time.perf_counter()
        try:
            if m == "cross":
                scores = self._cross_scores(question, docs)
                order = sorted(range(len(head)), key=lambda i: -scores[i])
                n_rel = sum(1 for s in scores if s >= float(getattr(C, "RERANK_MIN", 0.5)))
                top = scores[order[0]]
                ranked = [round(scores[i], 4) for i in order]
            elif m == "llm":
                picks = self._judge(question, docs)
                if picks is None:
                    self.failures += 1
                    return None
                order = picks + [i for i in range(len(head)) if i not in picks]
                n_rel = len(picks)
                top = 1.0 if picks else 0.0
                ranked = [1.0 if i in picks else 0.0 for i in order]
            else:
                return None
        except Exception:
            self.failures += 1
            self._fail_until = time.time() + 30.0
            return None
        self.calls += 1
        verdict = n_rel > 0
        public = {"mode": m, "verdict": verdict, "n_relevant": n_rel, "top_score": round(float(top), 4),
                  "scores": ranked, "ms": round((time.perf_counter() - t0) * 1000)}
        return {"mode": m, "candidates": [head[i] for i in order] + tail, "verdict": verdict,
                "n_relevant": n_rel, "public": public}
