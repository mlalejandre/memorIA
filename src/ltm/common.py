# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any

import config as _cfg

LEVELS = ["planeta", "pais", "ciudad", "fabrica", "persona", "organo", "tejido"]
N = len(LEVELS)

LEVELS_TEXT = (
    "La memoria es un árbol de profundidad VARIABLE (de 1 a 7 niveles), de lo más general a lo más concreto. "
    "Cada nodo es un CONCEPTO concreto con nombre propio y significativo (por ejemplo 'Perros', 'Riñón', 'España', "
    "'Preferencias de comunicación'). Cada célula (un único dato atómico) cuelga del nodo MÁS ESPECÍFICO que haga falta, "
    "sin ramas de relleno."
)

SYSTEM = (
    "Eres el gestor de memoria de un agente. " + LEVELS_TEXT +
    " Llamas 'el agente' al sistema y 'el usuario' a la persona que habla (su dueño). "
    "Responde SIEMPRE y SOLO con un JSON válido, sin texto adicional ni comentarios."
)

GENERIC_NODE_SLUGS = frozenset({
    "planeta", "pais", "ciudad", "fabrica", "organo", "tejido", "celula", "nodo", "general", "generales",
    "misc", "miscelanea", "otros", "otro", "varios", "sin-clasificar", "sin-categoria", "categoria", "tema", "area",
})

_ASK_RE = re.compile(
    r"^\W*(?:qué|cómo|quién|quiénes|cuándo|dónde|cuál|cuáles|cuánto|cuántos|por qué|"
    r"que (?:es|son|significa|hora)|como (?:se|puedo|es|funciona|hago)|donde (?:vivo|esta|estan)|cuando (?:es|fue)|"
    r"dime|dame|resume|resúmeme|resumeme|explica|explícame|explicame|cuéntame|cuentame|ayúdame|ayudame|"
    r"muéstrame|muestrame|enséñame|ensename|busca|búscame|buscame|hazme|escribe|traduce|calcula)\b", re.I)

_CHITCHAT_RE = re.compile(
    r"^\W*(?:hola|buenas|buenos días|buenos dias|gracias|vale|ok|okay|perfecto|genial|de acuerdo|adiós|adios|"
    r"hasta luego|entendido)\b", re.I)


def _is_nonfactual(text: str) -> bool:
    t = (text or "").strip()
    if _ASK_RE.match(t):
        return True
    return len(t) <= 30 and bool(_CHITCHAT_RE.match(t))


def slugify(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40].strip("-")
    return s or "general"


def extract_json(text: str) -> Any:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).replace("```json", "```")
    dec = json.JSONDecoder(strict=False)
    for i, ch in enumerate(text):
        if ch not in ("{", "["):
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, (dict, list)):
                return obj
        except ValueError:
            continue
    raise ValueError("no hay JSON en la respuesta")


def clamp(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def parse_answer(raw: str) -> dict[str, Any]:
    raw = raw or ""
    if not raw or "No tengo ese dato en memoria" in raw or "aún se está conectando" in raw:
        return {"found": False, "answer": "", "sources": []}
    
    sources = []
    ans_lines = []
    in_sources = False
    for line in raw.splitlines():
        sline = line.strip()
        if not sline:
            continue
        if sline.startswith("📚 Trazabilidad") or sline.startswith("📚 Fuentes"):
            in_sources = True
            continue
        if in_sources:
            if sline.startswith("•"):
                sources.append(sline.lstrip("• ").strip())
            elif sline.startswith("(") and sline.endswith(")"):
                in_sources = False
        elif sline.startswith("(") and sline.endswith(")"):
            continue
        else:
            if sline.startswith("💬"):
                ans_lines.append(sline.lstrip("💬 ").strip())
            elif not in_sources:
                ans_lines.append(sline)
                
    ans = "\n".join(ans_lines).strip()
    return {"found": bool(ans), "answer": ans, "sources": sources}


def clean_path_specs(specs: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in (specs if isinstance(specs, list) else []):
        if not isinstance(s, dict):
            s = {"name": str(s)}
        name = str(s.get("name") or "").strip()
        desc = str(s.get("description") or "").strip()
        if not name or slugify(name) in GENERIC_NODE_SLUGS:
            words = desc.replace(".", " ").split()
            if 0 < len(words) <= 4 and slugify(desc) not in GENERIC_NODE_SLUGS:
                s = {"name": desc.strip(" .,;:")[:60], "description": ""}
            else:
                continue
        if out and slugify(out[-1]["name"]) == slugify(s.get("name", "")):
            continue
        out.append(s)
    return out


def rrf_weight(channel: str) -> float:
    key = ("RRF_W_SYN" if channel.startswith("sinapsis") else
           "RRF_W_EXACT" if channel.startswith("exacta") else
           "RRF_W_FTS" if channel.startswith("léxica") else
           "RRF_W_EMB" if channel.startswith("embeddings") else
           "RRF_W_SEM" if channel.startswith("semántica") else
           "RRF_W_TREE" if channel.startswith("árbol") else "")
    if not key:
        return 1.0
    try:
        return float(getattr(_cfg, key, 1.0))
    except (TypeError, ValueError):
        return 1.0
