# -*- coding: utf-8 -*-
"""descomponer.py — Descomposición de preguntas en subconsultas de búsqueda.

Por qué: en LoCoMo, las preguntas multi-salto ("¿qué actividades ha hecho X?", "¿qué tienen en común X e Y?")
necesitan evidencia repartida en sesiones distintas. Con UNA sola consulta, la mitad de esa evidencia queda más
allá del puesto 50 (parche 15). Aquí el LLM reescribe la pregunta en 0-3 consultas breves, cada una sobre un único
dato; la recuperación busca con la pregunta original y con cada subconsulta y FUSIONA las listas (RRF) sin
aumentar el presupuesto (los mismos K hechos y E episodios llegan al prompt).

Si la pregunta busca un solo dato, el LLM devuelve una lista vacía y la recuperación es idéntica a la actual.
Desactivado por defecto (LTM_DECOMPOSE=0) hasta que un experimento pareado lo justifique.
"""
from __future__ import annotations

import re
import time
from typing import Any

import config as C

DECOMP_SYSTEM = ("Eres el planificador de búsqueda de una memoria a largo plazo. "
                 "Responde SIEMPRE y SOLO con un JSON válido, sin texto adicional.")

# Prompt GENERAL (sin ejemplos sacados de LoCoMo ni del corpus propio).
DECOMP_PROMPT = (
    "Pregunta: {q}\n\n"
    "Vas a buscar en una memoria de conversaciones pasadas para responderla. Decide si basta con una búsqueda.\n"
    "- Si la respuesta es un único dato, devuelve una lista vacía.\n"
    "- Si exige reunir VARIOS datos distintos (varios elementos de una lista, hechos de momentos diferentes, "
    "datos de varias personas o un dato intermedio para llegar a otro), escribe de 2 a {n} consultas de búsqueda "
    "breves, cada una sobre UN solo dato.\n"
    "Reglas: conserva los nombres propios en cada consulta; escribe en el mismo idioma que la pregunta; no "
    "respondas la pregunta ni añadas datos que no estén en ella.\n"
    'Devuelve SOLO este JSON: {{"consultas": ["...", "..."]}}'
)


def enabled() -> bool:
    return bool(getattr(C, "LTM_DECOMPOSE", False))


def max_subqueries() -> int:
    return max(1, min(5, int(getattr(C, "LTM_DECOMPOSE_MAX", 3))))


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", str(text).lower()).strip()


def clean(question: str, raw: Any, n: int | None = None, min_n: int = 2) -> list[str]:
    """Valida la salida del LLM: lista de cadenas, sin vacías, sin duplicados ni copias de la pregunta."""
    n = n or max_subqueries()
    items = raw.get("consultas") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    seen = {_norm(question)}
    out: list[str] = []
    for it in items:
        s = re.sub(r"\s+", " ", str(it if not isinstance(it, dict) else it.get("consulta", ""))).strip()
        if len(s) < 3 or len(s) > 200:
            continue
        k = _norm(s)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= n:
            break
    return out if len(out) >= min_n else []   # una sola subconsulta no es una descomposición min_n


def decompose(llm: Any, question: str) -> tuple[list[str], float]:
    """Devuelve (subconsultas, segundos). Ante cualquier fallo, ([], t): la recuperación sigue como siempre."""
    t0 = time.perf_counter()
    question = (question or "").strip()
    if not question or llm is None:
        return [], 0.0
    try:
        obj = llm.ask_json(DECOMP_SYSTEM, DECOMP_PROMPT.format(q=question, n=max_subqueries()))
    except Exception:
        return [], time.perf_counter() - t0
    return clean(question, obj), time.perf_counter() - t0


def fuse_many(lists: list[list[dict[str, Any]]], limit: int, k: int = 60, key: str = "id") -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion de N listas (la primera, la pregunta original). Conserva la mayor similitud
    ('sim' / 'score') que tuvo cada elemento en cualquiera de las listas."""
    lists = [lst for lst in lists if lst]
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0][:limit])
    scores: dict[Any, float] = {}
    by_id: dict[Any, dict[str, Any]] = {}
    best: dict[Any, dict[str, float]] = {}
    for lst in lists:
        for rank, it in enumerate(lst):
            i = it[key]
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(i, dict(it))
            for f in ("sim", "score"):
                if it.get(f) is not None:
                    v = float(it[f])
                    if v > best.setdefault(i, {}).get(f, float("-inf")):
                        best[i][f] = v
    order = sorted(scores, key=lambda i: -scores[i])[:limit]   # estable: a igualdad, manda la pregunta original
    out = []
    for i in order:
        it = by_id[i]
        for f, v in best.get(i, {}).items():
            it[f] = v
        out.append(it)
    return out
