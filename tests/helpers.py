# -*- coding: utf-8 -*-
# Simuladores deterministas (LLM y embeddings) compartidos por los tests y el benchmark emulado.
from __future__ import annotations

import hashlib
import math
import re
import sys
import unicodedata
from array import array
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ltm.common import slugify  # noqa: E402


def fold(text: Any) -> str:
    t = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


# ==============================================================================
# 1. SIMULADORES (solo modo emulado)
# ==============================================================================

_STOP = frozenset(fold(w) for w in (
    "el la los las un una unos unas de del al en y o a que es se por con para su sus lo como cual cuales "
    "qué quién dónde cuándo cómo cuál cuántos cuánto esta este esto ese esa eso son fue ser hay mi mis tu"
).split())


class MockEmbedder:
    """Embeddings deterministas por n-gramas de caracteres: capturan solapamiento léxico/morfológico,
    NO sinonimia. Sirven para validar la fontanería vectorial, no la calidad semántica."""
    model = "mock-ngram-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.calls = 0

    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[array]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> array:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", fold(text)):
            if w in _STOP or len(w) < 2:
                continue
            w = w[:7]
            grams = [w] if len(w) < 4 else [w[i:i + 3] for i in range(len(w) - 2)]
            for g in grams:
                h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
                v[h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return array("f", (x / n for x in v))


class MockLLM:
    """LLM simulado compatible con MemoryTree/ShortTermMemory. Decide con tablas que rellena el arnés."""

    def __init__(self) -> None:
        self.calls = 0
        self.url = "http://mock-llm/v1"
        self.last_finish_reason = "stop"
        self.domain_of: dict[str, str] = {}
        self.attr_of: dict[str, str] = {}

    def register(self, text: str, domain: str, attr: str | None = None) -> None:
        self.domain_of[_key(text)] = domain
        if attr:
            self.attr_of[_key(text)] = attr

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text or "")) // 4)

    def chat(self, messages, max_tokens=None, temperature=None) -> str:
        self.calls += 1
        content = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        content = content.split("FRAGMENTO:")[-1]
        words = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúñÑ0-9\-]{5,}", content)
        return "Resumen sintético: " + ", ".join(list(dict.fromkeys(words))[:30]) + "."

    def ask(self, system: str, user: str) -> str:
        self.calls += 1
        return ""

    def ask_json(self, system: str, user: str, attempts: int = 2) -> Any:
        self.calls += 1
        if "CLASIFICA LA RELACIÓN TEMPORAL" in user:
            mo = re.search(r"Hecho previo \(id=[^)]*\): (.*)", user)
            mn = re.search(r"Hecho nuevo \(fecha[^)]*\): (.*)", user)
            a = self.attr_of.get(_key(mo.group(1))) if mo else None
            b = self.attr_of.get(_key(mn.group(1))) if mn else None
            if a and a == b:
                return {"relation": "supersedes", "confidence": 0.95, "reason": "mismo atributo"}
            return {"relation": "complementary", "confidence": 0.9, "reason": "atributos distintos"}
        if "Decide si basta con una búsqueda" in user:  # [PARCHE19] descomposición: parte por " and " / " y "
            m = re.search(r"Pregunta: (.*)", user)
            q = (m.group(1) if m else "").strip().rstrip("?")
            parts = [p.strip() for p in re.split(r"\band\b|\by\b", q) if len(p.strip()) >= 3]
            return {"consultas": parts if len(parts) >= 2 else []}
        if "Dato a memorizar:" in user:
            m = re.search(r"Dato a memorizar: (.*)", user)
            dom = self.domain_of.get(_key(m.group(1)), "General") if m else "General"
            md = re.search(r"Profundidad actual: (\d+)", user)
            if md and int(md.group(1)) >= 1:
                return {"use": None, "here": True}
            if ("- " + slugify(dom) + " (") in user:
                return {"use": slugify(dom), "update_description": None}
            return {"use": None, "path": [{"name": dom, "description": f"Dominio {dom}"}]}
        if "palabras clave, sinónimos" in user:
            if "dedica" in user:
                return {"terminos": ["trabajo", "profesor", "empleo", "instituto"]}
            return {"terminos": []}
        if "Analiza el siguiente texto" in user:
            m = re.search(r"<<<(.*?)>>>", user, re.S)
            return {"hechos": [{"texto": (m.group(1) if m else user)[:200].strip(), "importancia": 7, "tipo": "dato"}]}
        if "Formula una pregunta" in user:
            m = re.search(r'este dato: "(.*)"\.', user, re.S)
            words = re.findall(r"\w{5,}", m.group(1)) if m else []
            return {"pregunta": " ".join(words[:5]) or "¿Cuál es el dato registrado?"}
        if "Escribe una descripción del ÁMBITO" in user:
            m = re.search(r"Nodo '(.+?)'", user)
            return {"description": f"Ámbito de {m.group(1) if m else 'nodo'}"}
        if "Compara estos dos recuerdos" in user:
            return {"redundant": False}
        if "¿Pertenecen estos dos subtemas" in user:
            return {"mergeable": False}
        if "Evalúa si este conocimiento" in user:
            return {"preserve": False}
        return {}
