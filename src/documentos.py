# -*- coding: utf-8 -*-
"""Ingesta de DOCUMENTOS con hechos autocontenidos (DOC_SELF_CONTAINED).

Problema medido en el parche 27 (cartucho «santander»): al trocear un artículo en hechos atómicos, algunos hechos
pierden su contexto («El crucero Canarias arribó al puerto…» sin decir qué noche ni tras qué suceso; «Se procedió a
la incautación de las tejeras…» sin decir tras qué) y el agente los atribuye al suceso equivocado.

Con DOC_SELF_CONTAINED=1 la ingesta de documentos (panel de ingesta de la consola) pide además que cada hecho se
entienda SOLO: nombra su suceso o sujeto y la fecha absoluta cuando el texto permite deducirla, y recibe el título
del documento y el final del fragmento anterior como contexto. La ingesta de conversaciones NO cambia (no pasa por
aquí; LoCoMo sigue midiendo el mismo camino).
"""
from __future__ import annotations

from typing import Any

import config as C
from ltm.common import SYSTEM, clamp

RULE = (
    "6. DOCUMENTO: cada hecho debe entenderse SOLO, sin el resto del texto. Nombra explícitamente el suceso, la "
    "persona, el lugar o la cosa de la que habla (nunca «el barco», «el incendio», «la ciudad» o «esa noche» sin decir "
    "cuál) y, cuando el texto permita deducirla, escribe la fecha absoluta (día, mes y año) en lugar de expresiones "
    "relativas como «esa misma noche», «el día 18», «una hora después» o «al día siguiente». No inventes fechas ni "
    "datos que el texto no permita deducir. Si una frase compara o encadena dos momentos, conserva el orden exacto "
    "(«una hora después de saberse la carga» no es «una hora después de la primera explosión»).")


def enabled() -> bool:
    return bool(getattr(C, "DOC_SELF_CONTAINED", False))


def prompt(chunk: str, source: str = "", previous: str = "") -> str:
    parts = [
        "Analiza el siguiente fragmento de un DOCUMENTO para incorporarlo al árbol de conocimiento de memoria.",
    ]
    if source:
        parts.append(f"Documento: {source}")
    if previous:
        parts.append("Final del fragmento anterior (solo para resolver referencias; NO extraigas hechos de aquí):")
        parts.append("«" + previous[-600:] + "»")
    parts += [
        "<<<" + str(chunk)[:6000] + ">>>",
        "",
        "Extrae TODOS los hechos, principios, conceptos clave o datos sustanciales y duraderos.",
        "Reglas fundamentales:",
        "1. Hechos personales: si habla del usuario, redáctalo en tercera persona: 'El usuario...'.",
        "2. Conocimiento técnico/académico: cada concepto como afirmación atómica, autosuficiente y universal.",
        "3. Ignora saludos, preguntas, órdenes, relleno conversacional, botones web, enlaces y números de página. "
        'Si no hay nada que memorizar devuelve "hechos": []. Cada hecho debe contener el dato CONCRETO (nombre, '
        "lugar, cifra); nunca hechos vagos.",
        "4. Importancia (1 a 10): 10 = identidad o directivas fundamentales; 7-9 = principios clave; 4-6 = secundarios; 1-3 = menores.",
        "5. Tipo: 'dato' o 'instruccion'.",
        RULE,
        'Devuelve EXCLUSIVAMENTE este JSON: {"hechos":[{"texto":"...","importancia":8,"tipo":"dato"}]}',
    ]
    return "\n".join(parts)


def parse(obj: Any) -> list[dict[str, Any]] | None:
    if not isinstance(obj, dict):
        return None
    out: list[dict[str, Any]] = []
    raw = obj.get("hechos")
    if isinstance(raw, list):
        for f in raw:
            if isinstance(f, str):
                f = {"texto": f}
            if not isinstance(f, dict):
                continue
            t = str(f.get("texto", "")).strip()
            if len(t) < 3:
                continue
            out.append({"texto": t[:600], "importancia": clamp(f.get("importancia", 6), 1, 10, 6),
                        "tipo": "instruccion" if f.get("tipo") == "instruccion" else "dato"})
    return out


def extract(llm: Any, chunk: str, source: str = "", previous: str = "") -> list[dict[str, Any]] | None:
    """Hechos autocontenidos de un fragmento (None si el LLM falla: el llamador cae a la ingesta normal)."""
    try:
        return parse(llm.ask_json(SYSTEM, prompt(chunk, source, previous)))
    except Exception:
        return None
