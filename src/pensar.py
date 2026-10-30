# -*- coding: utf-8 -*-
"""[PARCHE24] Pensamiento opcional del agente.

Modos (AGENT_THINK en .env, o por mensaje desde la interfaz):
  off   el agente responde directamente (defecto; ~1,8 s por respuesta en el servidor de casa)
  on    piensa SIEMPRE antes de responder (~5 veces más lento; P23: +3,5 pp en LoCoMo sin significación)
  auto  el agente decide por pregunta: piensa solo cuando la pregunta pide CALCULAR algo con lo recordado
        (fechas y plazos, recuentos, combinar datos de dos personas o momentos), que es donde el pensamiento
        ayudó en P23; las preguntas directas y las de inferencia (donde pensar le volvió prudente de más) van sin
        pensar. Reglas transparentes: cada decisión lleva su motivo y la interfaz lo muestra.

Solo afecta a la respuesta final del agente: la ingesta, la recuperación y los jueces no cambian.
"""
from __future__ import annotations

import re

import config as C

MODES = ("off", "auto", "on")
_ALIASES = {"0": "off", "no": "off", "false": "off", "off": "off", "nunca": "off",
            "1": "on", "si": "on", "sí": "on", "yes": "on", "true": "on", "on": "on", "siempre": "on",
            "auto": "auto", "automatico": "auto", "automático": "auto"}

# (motivo, patrón) en orden de prioridad · español e inglés (LoCoMo está en inglés; el usuario escribe en español)
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fechas", re.compile(
        r"\b(when|what (date|day|month|year|time)|which (date|day|month|year|week)|how long|"
        r"how many (days|weeks|months|years|hours)|how old|since when|ago|"
        r"cu[aá]ndo|qu[eé] (d[ií]a|fecha|mes|a[nñ]o|semana)|en qu[eé] (fecha|mes|a[nñ]o|d[ií]a|semana)|"
        r"hace cu[aá]nto|cu[aá]nto tiempo|desde cu[aá]ndo|cu[aá]nt[oa]s (d[ií]as|meses|a[nñ]os|semanas|horas)|"
        r"qu[eé] edad)\b", re.I)),
    ("recuento", re.compile(
        r"\b(how many|how often|number of|in total|cu[aá]nt[oa]s|cu[aá]ntas veces|con qu[eé] frecuencia|"
        r"en total)\b", re.I)),
    ("combinar datos", re.compile(
        r"\b(both|in common|each other|compare|comparison|difference between|ambos|ambas|los dos|las dos|"
        r"en com[uú]n|compar[ae]|comparando|diferencia entre)\b", re.I)),
)


def normalize(mode: str | None) -> str:
    return _ALIASES.get(str(mode or "").strip().lower(), "off")


def configured_mode() -> str:
    return normalize(getattr(C, "AGENT_THINK", "off"))


def decide(question: str, mode: str | None = None) -> tuple[bool, str]:
    """(pensar, motivo). mode None = el de la configuración."""
    m = configured_mode() if mode is None else normalize(mode)
    if m == "off":
        return False, "desactivado"
    if m == "on":
        return True, "activado siempre"
    q = str(question or "")
    for reason, pat in RULES:
        if pat.search(q):
            return True, reason
    return False, "pregunta directa"


def extra_tokens() -> int:
    """Tokens que se SUMAN al límite de la respuesta cuando piensa (el razonamiento cuenta en max_tokens)."""
    return max(512, int(getattr(C, "AGENT_THINK_EXTRA_TOKENS", 4096)))
