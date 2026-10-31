# -*- coding: utf-8 -*-
"""refinar.py — Dos formas de reunir evidencia repartida entre sesiones, con el MISMO presupuesto.

Motivo (parche 19): el multi-salto de LoCoMo casi nunca son preguntas sobre varias personas, sino sobre UNA persona
con la evidencia repartida en varias sesiones ("What did Caroline research?"). Partir la pregunta no ayuda (el LLM
solo la partió en el 9 % de los casos y la cobertura no cambió), pero buscar con la respuesta (oráculo) sube la
cobertura completa del 30,5 % al 59,2 %: la evidencia está en la memoria y las consultas no la alcanzan.

  1. Diversidad por día (EPI_DIVERSITY=dia, sin LLM): los episodios candidatos se buscan más hondo y, al elegir los
     que llegan al prompt, se admiten como mucho EPI_PER_DAY por día. Hipótesis: los turnos vecinos de UNA sesión
     se parecen entre sí y acaparan los 20 huecos. Si no hay bastantes días distintos, se rellena en orden.
  2. Segunda pasada (LTM_SECOND_PASS=1, 1 llamada LLM): el LLM lee lo recuperado en la primera búsqueda y propone
     1-3 consultas con términos CONCRETOS de lo encontrado para buscar lo que falte (otros momentos, otros elementos
     de una lista). Es lo más parecido al oráculo que existe sin conocer la respuesta. Lo recuperado con ellas se
     fusiona (RRF) con la pregunta original, sin subir el presupuesto.
Ambas desactivadas por defecto hasta que un experimento pareado lo justifique.
"""
from __future__ import annotations

import time
from typing import Any

import config as C
import descomponer

REFINE_SYSTEM = ("Eres el planificador de búsqueda de una memoria a largo plazo. "
                 "Responde SIEMPRE y SOLO con un JSON válido, sin texto adicional.")

# Prompt GENERAL (sin ejemplos de LoCoMo ni del corpus propio).
REFINE_PROMPT = (
    "Pregunta: {q}\n\n"
    "Esto es lo que ha encontrado una primera búsqueda en la memoria de conversaciones pasadas:\n"
    "{ctx}\n\n"
    "Puede que la respuesta completa necesite recuerdos que todavía no han aparecido: otras ocasiones, otros "
    "elementos de una lista, o el dato que falta para responder. Escribe de 1 a {n} consultas de búsqueda breves y "
    "concretas para encontrarlos, usando los nombres propios y los términos específicos que aparecen arriba. Si lo "
    "encontrado ya basta para responder por completo, devuelve una lista vacía.\n"
    "Reglas: escribe en el mismo idioma que la pregunta; no respondas la pregunta; no inventes datos.\n"
    'Devuelve SOLO este JSON: {{"consultas": ["..."]}}'
)


def second_pass_enabled() -> bool:
    return bool(getattr(C, "LTM_SECOND_PASS", False))


def diversity_mode() -> str:
    return str(getattr(C, "EPI_DIVERSITY", "off") or "off").strip().lower()


def per_day() -> int:
    return max(1, int(getattr(C, "EPI_PER_DAY", 3)))


def day_of(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def por_dia(items: list[dict[str, Any]], limit: int, cap: int | None = None) -> list[dict[str, Any]]:
    """Selecciona `limit` elementos en el orden dado, con como mucho `cap` del mismo día; si no hay bastantes días
    distintos, rellena con los descartados en su orden original."""
    cap = cap or per_day()
    out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    count: dict[str, int] = {}
    for it in items:
        if len(out) >= limit:
            break
        d = day_of(it.get("ts"))
        if count.get(d, 0) < cap:
            out.append(it)
            count[d] = count.get(d, 0) + 1
        else:
            skipped.append(it)
    if len(out) < limit:
        out += skipped[:limit - len(out)]
    return out


def contexto(fact_lines: list[str], episodes: list[dict[str, Any]], n_facts: int = 10, n_epi: int = 10,
             width: int = 220) -> str:
    """Texto breve con lo recuperado en la primera búsqueda (hechos y episodios), para la segunda pasada."""
    lines = [f"- {str(f).strip()[:width]}" for f in fact_lines[:n_facts] if str(f).strip()]
    lines += [f"- {str(e.get('text', '')).strip()[:width]}" for e in episodes[:n_epi] if str(e.get("text", "")).strip()]
    return "\n".join(lines) or "(nada)"


def clean(question: str, raw: Any, n: int | None = None) -> list[str]:
    """Como descomponer.clean, pero una única consulta ya es válida (busca un dato que falta)."""
    return descomponer.clean(question, raw, n or descomponer.max_subqueries(), min_n=1)


def second_pass(llm: Any, question: str, ctx: str) -> tuple[list[str], float]:
    """Devuelve (consultas, segundos). Ante cualquier fallo, ([], t): la recuperación queda como en la 1.ª pasada."""
    t0 = time.perf_counter()
    question = (question or "").strip()
    if not question or llm is None:
        return [], 0.0
    try:
        obj = llm.ask_json(REFINE_SYSTEM, REFINE_PROMPT.format(q=question, ctx=ctx, n=descomponer.max_subqueries()))
    except Exception:
        return [], time.perf_counter() - t0
    return clean(question, obj), time.perf_counter() - t0
