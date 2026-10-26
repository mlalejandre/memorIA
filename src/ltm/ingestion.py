# -*- coding: utf-8 -*-
from __future__ import annotations

import queue
import threading
import time
from typing import Any

import config as C
from .common import SYSTEM, clamp, _is_nonfactual


class IngestionMixin:
    """Cola de ingesta en segundo plano y extracción de hechos con LLM."""

    def _init_ingest_queue(self) -> None:
        self._ingest_q: queue.Queue = queue.Queue(maxsize=max(1, C.LTM_INGEST_QUEUE_MAX))
        self._ingest_thread: threading.Thread | None = None
        self._ingest_stop = threading.Event()
        self._ingest_guard = threading.Lock()

    def extract_facts(self, text: str, first: bool = False, origin: str = "") -> tuple[str | None, list[dict[str, Any]] | None]:
        parts = [
            "Analiza el siguiente texto para incorporarlo al árbol de conocimiento de memoria:",
            "<<<" + str(text)[:6000] + ">>>",
            "",
            "Extrae TODOS los hechos, principios, conceptos clave o datos sustanciales y duraderos.",
            "Reglas fundamentales:",
            "1. Hechos personales: si habla del usuario, redáctalo en tercera persona: 'El usuario...'.",
            "2. Conocimiento técnico/académico: cada concepto como afirmación atómica, autosuficiente y universal.",
            "3. Ignora saludos, preguntas, órdenes, relleno conversacional, botones web, enlaces y números de página. "
            'Si no hay nada que memorizar, o el texto es una pregunta, una orden o una petición, devuelve "hechos": []. '
            'Cada hecho debe contener el dato CONCRETO (nombre, lugar, cifra); nunca hechos vagos como "vive en una ubicación específica".',
            "4. Importancia (1 a 10): 10 = identidad o directivas fundamentales; 7-9 = principios clave; 4-6 = secundarios; 1-3 = menores.",
            "5. Tipo: 'dato' o 'instruccion'.",
        ]
        if "agente" in origin.lower():
            parts.append("6. El texto es una respuesta del agente: registra solo conocimiento verificable, no opiniones ni relleno.")
        if first:
            parts.append("7. Si las primeras líneas son cabecera, título o autores, sintetízalo en 'fuente_detectada'; si no, null.")
            parts.append('Devuelve EXCLUSIVAMENTE este JSON: {"fuente_detectada": "Título y Autores o null", "hechos":[{"texto":"...","importancia":8,"tipo":"dato"}]}')
        else:
            parts.append('Devuelve EXCLUSIVAMENTE este JSON: {"hechos":[{"texto":"...","importancia":8,"tipo":"dato"}]}')
        try:
            obj = self.llm.ask_json(SYSTEM, "\n".join(parts))
        except Exception:
            return None, None
        if not isinstance(obj, dict):
            return None, None
        facts: list[dict[str, Any]] = []
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
                facts.append({
                    "texto": t[:600],
                    "importancia": clamp(f.get("importancia", 6), 1, 10, 6),
                    "tipo": "instruccion" if f.get("tipo") == "instruccion" else "dato",
                })
        src = obj.get("fuente_detectada")
        src = str(src).strip() if src else ""
        return (src if src and src.lower() not in ("null", "none") else None), facts

    def ingest_async(self, text: str, origin: str = "") -> bool:
        text = (text or "").strip()
        if len(text) < C.LTM_MIN_CHARS:
            return False
        if text.endswith("?") or text.endswith("¿") or _is_nonfactual(text):
            return False
        is_agent = "agente" in origin.lower()
        if (is_agent and not C.LTM_INGEST_ASSISTANT) or (not is_agent and not C.LTM_INGEST_USER):
            return False

        when_str = time.strftime("%d/%m/%Y a las %H:%M")
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM ingest_queue WHERE status = 'pending'")
            pending = cur.fetchone()[0]
            if pending >= C.LTM_INGEST_QUEUE_MAX:
                self.stats["ingest_dropped"] = self.stats.get("ingest_dropped", 0) + 1
                return False
            cur.execute(
                "INSERT INTO ingest_queue (text, origin, timestamp, status, created_tick) VALUES (?, ?, ?, 'pending', ?)",
                (text[:6000], origin or "conversación", when_str, self.tick)
            )
            item_id = cur.lastrowid
            self.conn.commit()

        try:
            self._ingest_q.put_nowait((item_id, text[:6000], origin or "conversación", when_str))
        except queue.Full:
            pass

        self._ensure_ingest_worker()
        return True

    def _ensure_ingest_worker(self) -> None:
        with self._ingest_guard:
            if self._ingest_thread and self._ingest_thread.is_alive():
                return
            self._ingest_stop.clear()
            self._ingest_thread = threading.Thread(target=self._ingest_loop, name="ltm-ingest", daemon=True)
            self._ingest_thread.start()

    def _ingest_loop(self) -> None:
        while not self._ingest_stop.is_set():
            from_queue = True
            try:
                item = self._ingest_q.get(timeout=1.0)
            except queue.Empty:
                from_queue = False
                with self.lock:
                    cur = self.conn.cursor()
                    cur.execute("SELECT id, text, origin, timestamp FROM ingest_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1")
                    r = cur.fetchone()
                    if r:
                        item = (r["id"], r["text"], r["origin"], r["timestamp"])
                    else:
                        continue

            item_id, text, origin, when = item
            with self.lock:
                self.conn.execute("UPDATE ingest_queue SET status = 'processing' WHERE id = ?", (item_id,))
                self.conn.commit()

            try:
                source = f"Conversación con el usuario ({origin}) el {when}"
                _, facts = self.extract_facts(text, first=False, origin=origin)
                for f in facts or []:
                    imp = max(9.0, f["importancia"]) if f["tipo"] == "instruccion" else f["importancia"]
                    self.store(f["texto"], imp, f["tipo"], meta={"fuente": source})
                    self.stats["ingest_facts"] = self.stats.get("ingest_facts", 0) + 1
                with self.lock:
                    self.conn.execute("UPDATE ingest_queue SET status = 'done' WHERE id = ?", (item_id,))
                    self.conn.commit()
            except Exception:
                with self.lock:
                    self.conn.execute("UPDATE ingest_queue SET status = 'failed' WHERE id = ?", (item_id,))
                    self.conn.commit()
            finally:
                if from_queue:
                    self._ingest_q.task_done()
