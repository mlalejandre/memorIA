# -*- coding: utf-8 -*-
from __future__ import annotations

import config as C
from llm import LLM


class MemLLM(LLM):
    """Cliente LLM unificado para LTM y tareas de memoria neocortical.
    
    Hereda de LLM implementando streaming, conteo exacto de tokens por /tokenize
    y manejo robusto de excepciones de red, manteniendo retrocompatibilidad total.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        super().__init__(
            base_url=base_url or C.LLM_LTM_URL,
            model=model or C.REMOTE_MODEL_ALIAS
        )
        self.default_max_tokens = C.MEM_MAX_TOKENS
        self.default_temperature = 0.2

    def ask(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        return self.chat(
            messages,
            max_tokens=self.default_max_tokens,
            temperature=self.default_temperature
        )
