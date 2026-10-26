# Cliente OpenAI-compatible (sin streaming).
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import config as C


class LLM:
    """Cliente OpenAI-compatible (llama-server, vLLM, Ollama...) sin streaming."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or C.LLM_BASE_URL).rstrip("/")
        self.model = model or C.LLM_MODEL
        self.url = self.base_url + "/chat/completions"
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.tokenize_url = root + "/tokenize"
        self.calls = 0
        self._tok_fail_until = 0.0
        self._local = threading.local()

    @property
    def last_finish_reason(self) -> str | None:
        """finish_reason de la última respuesta de ESTE hilo ('stop', 'length'...)."""
        return getattr(self._local, "finish", None)

    def chat(self, messages: list[dict[str, str]], max_tokens: int | None = None,
             temperature: float | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": C.ANSWER_TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or C.ANSWER_MAX_TOKENS,
            "stream": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": C.THINKING},
            "messages": messages,
        }
        body = json.dumps(payload).encode("utf-8")
        data: Any = None
        last_exc: Exception | None = None
        for attempt in range(2):
            req = urllib.request.Request(
                self.url, data=body, headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=C.REQUEST_TIMEOUT_S) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                break
            except urllib.error.HTTPError as exc:
                err = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} del servidor LLM ({self.url}): {err[:400]}") from exc
            except Exception as exc:  # red caída / timeout: un único reintento
                last_exc = exc
                if attempt == 0:
                    time.sleep(1.0)
        if data is None:
            raise RuntimeError(
                f"Fallo de red/LLM ({self.url}): {type(last_exc).__name__}: {last_exc}") from last_exc
        self.calls += 1
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Respuesta LLM inesperada: {str(data)[:300]}") from exc
        self._local.finish = choice.get("finish_reason") if isinstance(choice, dict) else None
        text = (msg.get("content") or "").strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return text or (msg.get("reasoning_content") or "").strip()

    def _sse_deltas(self, resp: Any):
        """Itera una respuesta SSE de /chat/completions y devuelve los fragmentos de texto."""
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            choices = obj.get("choices") if isinstance(obj, dict) else None
            if not choices or not isinstance(choices[0], dict):
                continue
            fin = choices[0].get("finish_reason")
            if fin:
                self._local.finish = fin
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta

    def chat_stream(self, messages: list[dict[str, str]], max_tokens: int | None = None,
                    temperature: float | None = None):
        """Como chat(), pero devuelve el texto a medida que llega (generador de str).
        Descarta un bloque <think>...</think> inicial si el modelo lo emite."""
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": C.ANSWER_TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or C.ANSWER_MAX_TOKENS,
            "stream": True,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": C.THINKING},
            "messages": messages,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"}, method="POST",
        )
        self._local.finish = None
        try:
            resp = urllib.request.urlopen(req, timeout=C.REQUEST_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            err = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} del servidor LLM ({self.url}): {err[:400]}") from exc
        except Exception as exc:
            raise RuntimeError(f"Fallo de red/LLM ({self.url}): {type(exc).__name__}: {exc}") from exc
        self.calls += 1
        buf = ""
        mode = "probe"  # probe -> think -> strip_lead -> out
        with resp:
            for delta in self._sse_deltas(resp):
                if mode == "out":
                    yield delta
                    continue
                if mode == "strip_lead":
                    cleaned = delta.lstrip()
                    if cleaned:
                        mode = "out"
                        yield cleaned
                    continue
                buf += delta
                if mode == "probe":
                    probe = buf.lstrip()
                    if len(probe) < 7 and "<think>".startswith(probe):
                        continue
                    if probe.startswith("<think>"):
                        mode = "think"
                    else:
                        mode = "out"
                        yield buf
                        buf = ""
                        continue
                if mode == "think" and "</think>" in buf:
                    rest = buf.split("</think>", 1)[1].lstrip()
                    buf = ""
                    if rest:
                        mode = "out"
                        yield rest
                    else:
                        mode = "strip_lead"
        if mode == "probe" and buf:
            yield buf

    def count_tokens(self, text: str) -> int:
        """Tokens exactos vía /tokenize; si falla, estimación por caracteres."""
        if C.STM_EXACT_TOKENS and time.time() >= self._tok_fail_until:
            try:
                req = urllib.request.Request(
                    self.tokenize_url, data=json.dumps({"content": text}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    toks = json.loads(resp.read().decode("utf-8")).get("tokens")
                if isinstance(toks, list):
                    return max(1, len(toks))
            except Exception:
                self._tok_fail_until = time.time() + 30
        return max(1, int(len(text) / C.STM_CHARS_PER_TOKEN))

    def ask(self, system: str, user: str) -> str:
        """Compatibilidad con la interfaz LTM de MemLLM."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        return self.chat(messages)

    def ask_json(self, system: str, user: str, attempts: int = 2) -> Any:
        """Compatibilidad con extracción JSON en LTM."""
        from ltm.common import extract_json
        last = None
        for i in range(attempts):
            u = user if i == 0 else user + "\n\nResponde SOLO con el JSON, sin texto adicional."
            text = self.ask(system, u)
            try:
                return extract_json(text)
            except ValueError as exc:
                last = exc
        raise ValueError(f"El modelo no devolvió JSON válido: {last}")

