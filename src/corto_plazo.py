from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import config as C

_STOP = {
    "a", "al", "algo", "como", "con", "cual", "cuando", "de", "del", "el", "ella", "en", "es", "esa",
    "ese", "eso", "esta", "este", "esto", "fue", "ha", "hay", "la", "las", "le", "les", "lo", "los",
    "me", "mi", "mis", "muy", "no", "o", "para", "pero", "por", "que", "quien", "se", "ser", "si",
    "sin", "sobre", "su", "sus", "te", "tiene", "un", "una", "uno", "y", "yo", "tu", "tus", "the",
    "is", "are", "what", "and", "or", "dime", "dijiste", "dije",
}

SUMMARY_SYSTEM = (
    "Eres el módulo de memoria a corto plazo de un agente. Resumes fragmentos antiguos de una "
    "conversación de forma fiel, densa y sin inventar nada."
)


def _tok(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", str(text).lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return [(w[:-1] if len(w) > 4 and w.endswith("s") else w)[:7]
            for w in re.findall(r"[a-z0-9]+", t) if w not in _STOP and len(w) > 1]


def fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def role_label(e: dict[str, Any]) -> str:
    return {"user": "Usuario", "assistant": "Agente"}.get(e.get("role", ""), "Resumen")


class ShortTermMemory:
    """Memoria reciente de acceso rápido con autorresumen de lo más antiguo.

    entries: lista cronológica de dicts {id, role(user|assistant|summary), text, tokens, ts, level, ...}
    Cuando los tokens vivos se acercan a max_tokens, un hilo de fondo resume los fragmentos MÁS
    ANTIGUOS (los últimos `keep_recent` mensajes nunca se tocan) hasta bajar al nivel objetivo.
    Los originales resumidos se conservan en _archivo.jsonl para auditoría.
    """

    def __init__(self, llm: Any, path: Path | None = None, *, max_tokens: int | None = None,
                 trigger_ratio: float | None = None, target_ratio: float | None = None,
                 keep_recent: int | None = None, chunk_tokens: int | None = None,
                 summary_ratio: float | None = None, prompt_mode: str | None = None,
                 prompt_recent: int | None = None, lookup_k: int | None = None,
                 on_event: Callable[[str], None] | None = None) -> None:
        self.llm = llm
        self.dir = Path(path or C.STM_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens or C.STM_MAX_TOKENS
        self.trigger_tokens = int(self.max_tokens * (trigger_ratio or C.STM_TRIGGER_RATIO))
        self.target_tokens = int(self.max_tokens * (target_ratio or C.STM_TARGET_RATIO))
        self.keep_recent = C.STM_KEEP_RECENT if keep_recent is None else keep_recent
        self.chunk_tokens = chunk_tokens or C.STM_CHUNK_TOKENS
        self.summary_ratio = summary_ratio or C.STM_SUMMARY_RATIO
        self.prompt_mode = (prompt_mode or C.STM_PROMPT_MODE)
        self.prompt_recent = prompt_recent or C.STM_PROMPT_RECENT
        self.lookup_k = lookup_k or C.STM_LOOKUP_K
        self.on_event = on_event or (lambda m: None)

        self._lock = threading.RLock()
        self._compact_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.compacting = False
        self.entries: list[dict[str, Any]] = []
        self._seq = 0
        self.stats = {"lookups": 0, "compactions": 0, "summarized_entries": 0,
                      "tokens_saved": 0, "llm_summaries": 0, "extractive_summaries": 0}
        self._load()

    # ---------- persistencia ----------
    @property
    def _file(self) -> Path:
        return self.dir / "sesion.json"

    def _load(self) -> None:
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            try:  # no perder datos en silencio: conserva el archivo ilegible
                self._file.replace(self._file.with_name(f"sesion.json.corrupt-{int(time.time())}"))
            except OSError:
                pass
            return
        if isinstance(data, dict):
            self.entries = [e for e in data.get("entries", []) if isinstance(e, dict) and e.get("text")]
            self._seq = int(data.get("seq", len(self.entries)))
            if isinstance(data.get("stats"), dict):
                self.stats.update(data["stats"])

    def _save(self) -> None:
        tmp = self._file.with_name("sesion.json.tmp")
        tmp.write_text(json.dumps({"seq": self._seq, "entries": self.entries, "stats": self.stats},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self._file)

    def _archive(self, items: list[dict[str, Any]], replaced_by: str) -> None:
        try:
            with (self.dir / "_archivo.jsonl").open("a", encoding="utf-8") as fh:
                for e in items:
                    fh.write(json.dumps({"archived_at": time.time(), "replaced_by": replaced_by, **e},
                                        ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _emit(self, msg: str) -> None:
        try:
            self.on_event(msg)
        except Exception:
            pass

    # ---------- utilidades ----------
    def _count(self, text: str) -> int:
        return int(self.llm.count_tokens(text)) + C.STM_MSG_OVERHEAD

    def _total(self) -> int:
        return sum(int(e.get("tokens", 0)) for e in self.entries)

    def total_tokens(self) -> int:
        with self._lock:
            return self._total()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self.entries]

    # ---------- escritura ----------
    def add(self, role: str, text: str, ts: float | None = None) -> dict[str, Any] | None:
        text = str(text).strip()
        if not text:
            return None
        tokens = self._count(text)  # fuera del lock: puede hacer una llamada de red
        with self._lock:
            self._seq += 1
            entry = {"id": f"{self._seq:07d}", "role": role, "text": text, "tokens": tokens,
                     "ts": ts or time.time(), "level": 0}
            self.entries.append(entry)
            self._save()
            over = self._total() >= self.trigger_tokens
        if over:
            self._wake.set()
        return dict(entry)

    def reset(self) -> None:
        with self._lock:
            self.entries = []
            self._save()

    # ---------- lectura rápida (BM25 local, sin LLM) ----------
    def _bm25(self, query: str, docs: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        q = set(_tok(query))
        if not q or not docs:
            return []
        toks = [Counter(_tok(d["text"])) for d in docs]
        lens = [sum(c.values()) for c in toks]
        avg = (sum(lens) / len(lens)) or 1.0
        df: Counter[str] = Counter()
        for c in toks:
            df.update(c.keys())
        n, scored = len(docs), []
        for d, c, dl in zip(docs, toks, lens):
            s = 0.0
            for t in q:
                tf = c.get(t, 0)
                if not tf:
                    continue
                idf = math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * dl / avg))
            if s > 0:
                scored.append((s, d))
        scored.sort(key=lambda x: -x[0])
        return [{**d, "score": round(s, 3)} for s, d in scored[:k]]

    def recall(self, query: str) -> dict[str, Any]:
        """Parte rápida de la consulta: qué contexto reciente corresponde al prompt."""
        t0 = time.perf_counter()
        entries = self.snapshot()
        summaries = [e for e in entries if e["role"] == "summary"]
        raw = [e for e in entries if e["role"] != "summary"]
        if self.prompt_mode == "full":
            recent, older = raw, []
        else:
            recent = raw[-self.prompt_recent:]
            older = raw[:-self.prompt_recent] if len(raw) > self.prompt_recent else []
        hits = sorted(self._bm25(query, older, self.lookup_k), key=lambda e: e["ts"])
        with self._lock:
            self.stats["lookups"] += 1
        return {"summaries": summaries, "recent": recent, "hits": hits,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}

    # ---------- compactación autónoma ----------
    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="stm-compactor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(C.STM_COMPACT_POLL_S)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                if self.total_tokens() >= self.trigger_tokens:
                    self.compact()
            except Exception as exc:
                self._emit(f"[STM] error de compactación: {type(exc).__name__}: {exc}")

    def _pick_chunk(self) -> list[dict[str, Any]]:
        cands = self.entries[:-self.keep_recent] if self.keep_recent else list(self.entries)
        chunk: list[dict[str, Any]] = []
        total = 0
        for e in cands:
            chunk.append(e)
            total += int(e["tokens"])
            if total >= self.chunk_tokens:
                break
        if not chunk:
            return []
        if len(chunk) == 1 and chunk[0]["role"] == "summary" and chunk[0]["tokens"] <= C.STM_MIN_SUMMARY_TOKENS:
            return []  # un único resumen ya mínimo: no hay nada más que comprimir
        return [dict(e) for e in chunk]

    def compact(self, force: bool = False) -> list[str]:
        """Resume lo más antiguo hasta bajar al objetivo. force=True fuerza al menos una ronda."""
        reports: list[str] = []
        with self._compact_lock:
            self.compacting = True
            try:
                for i in range(C.STM_MAX_COMPACT_ROUNDS):
                    with self._lock:
                        total = self._total()
                        if i == 0:
                            if not force and total < self.trigger_tokens:
                                break
                        elif total <= self.target_tokens:
                            break
                        chunk = self._pick_chunk()
                    if not chunk:
                        reports.append("sin fragmentos antiguos que resumir")
                        break
                    rep = self._summarize_chunk(chunk)
                    reports.append(rep)
                    self._emit("[STM] " + rep)
            finally:
                self.compacting = False
        return reports

    def _render(self, chunk: list[dict[str, Any]]) -> str:
        lines = []
        for e in chunk:
            if e["role"] == "summary":
                lines.append(f"[RESUMEN PREVIO {fmt_ts(e.get('ts_from', e['ts']))} → {fmt_ts(e.get('ts_to', e['ts']))}] {e['text']}")
            else:
                lines.append(f"[{fmt_ts(e['ts'])}] {role_label(e)}: {e['text']}")
        return "\n".join(lines)

    def _extractive(self, chunk: list[dict[str, Any]], budget_tokens: int) -> str:
        per = max(60, int(budget_tokens * C.STM_CHARS_PER_TOKEN) // max(1, len(chunk)))
        parts = []
        for e in chunk:
            t = e["text"].replace("\n", " ")
            parts.append(f"{role_label(e)}: {t[:per]}{'…' if len(t) > per else ''}")
        return "(resumen extractivo) " + " | ".join(parts)

    def _summarize_chunk(self, chunk: list[dict[str, Any]]) -> str:
        ctok = sum(int(e["tokens"]) for e in chunk)
        budget = max(C.STM_MIN_SUMMARY_TOKENS, int(ctok * self.summary_ratio))
        words = max(40, int(budget * 0.7))
        user = (
            "Resume este fragmento de conversación (lo más antiguo primero). Conserva hechos, nombres, cifras, "
            "fechas, decisiones, preferencias del usuario, tareas pendientes y conclusiones. Elimina saludos, "
            "relleno y repeticiones. Si hay resúmenes previos, intégralos sin perder sus datos. "
            f"Escribe en español, en frases compactas, máximo {words} palabras. No inventes nada ni añadas "
            "comentarios.\n\nFRAGMENTO:\n" + self._render(chunk)
        )
        method, summary = "llm", ""
        try:
            summary = self.llm.chat(
                [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}],
                max_tokens=int(budget * 1.6) + 64, temperature=0.2,
            ).strip()
        except Exception as exc:
            self._emit(f"[STM] resumen LLM falló ({type(exc).__name__}); uso resumen extractivo")
        stoks = self._count(summary) if summary else 10 ** 9
        if not summary or stoks >= ctok * 0.9:
            summary, method = self._extractive(chunk, budget), "extractivo"
            stoks = self._count(summary)

        with self._lock:
            ids = {e["id"] for e in chunk}
            live = [e for e in self.entries if e["id"] in ids]
            if len(live) != len(chunk):
                return "fragmento modificado durante el resumen (¿reset?): se omite"
            pos = next(i for i, e in enumerate(self.entries) if e["id"] in ids)
            self._seq += 1
            new = {
                "id": f"{self._seq:07d}", "role": "summary", "text": summary, "tokens": stoks,
                "ts": max(e.get("ts_to", e["ts"]) for e in live),
                "ts_from": min(e.get("ts_from", e["ts"]) for e in live),
                "ts_to": max(e.get("ts_to", e["ts"]) for e in live),
                "level": max(int(e.get("level", 0)) for e in live) + 1,
                "covers": sum(int(e.get("covers", 1)) for e in live),
                "method": method,
            }
            self.entries = [e for e in self.entries if e["id"] not in ids]
            self.entries.insert(pos, new)
            self._archive(live, new["id"])
            self.stats["compactions"] += 1
            self.stats["summarized_entries"] += len(live)
            self.stats["tokens_saved"] += max(0, ctok - stoks)
            self.stats["llm_summaries" if method == "llm" else "extractive_summaries"] += 1
            self._save()
            total = self._total()
        return (f"{len(live)} mensajes/resúmenes ({ctok} tok) → resumen nivel {new['level']} "
                f"({stoks} tok, {method}). Memoria: {total}/{self.max_tokens} tok")

    # ---------- observación ----------
    def stats_snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self._total()
            return {
                "tokens": total, "max_tokens": self.max_tokens, "trigger": self.trigger_tokens,
                "target": self.target_tokens, "usage": round(total / max(1, self.max_tokens), 3),
                "entries": len(self.entries),
                "summaries": sum(1 for e in self.entries if e["role"] == "summary"),
                "compacting": self.compacting, "prompt_mode": self.prompt_mode, **self.stats,
            }
