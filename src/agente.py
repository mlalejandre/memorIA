# Agente con doble memoria (STM + LTM) consultadas en paralelo.
from __future__ import annotations

import platform
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import config as C
from cartuchos import CartridgeManager
from corto_plazo import ShortTermMemory, fmt_ts, role_label
from largo_plazo import MemoryTree, MemLLM
from llm import LLM

# [PARCHE6] regla anti-deducción: las 2 invenciones reales del parche 5 fueron deducciones de datos parecidos
BASE_PROMPT = (
    "Eres un asistente con DOS memorias consultadas en paralelo para cada interacción:\n"
    "- Memoria a corto plazo: resúmenes y mensajes de ESTA conversación.\n"
    "- Memoria a largo plazo: conocimiento consolidado y verificado, con fuentes.\n"
    "Responde en español combinando ambas con tus conocimientos. Prioriza el corto plazo para "
    "el contexto inmediato de la charla y la memoria larga para hechos consolidados. Cita las fuentes "
    "de la memoria larga si es relevante. Sobre el usuario y su entorno, afirma solo lo que conste de forma "
    "explícita en la memoria: si un dato personal no consta, dilo claramente en vez de deducirlo de otros "
    "datos parecidos. "
    # [PARCHE12] atribución y fechas relativas: en LoCoMo, 52 de 57 invenciones atribuían a la persona
    # preguntada lo que otra persona dijo de sí misma (42 de ellas, leyéndolo de la memoria episódica).
    "Atribución: cada recuerdo indica de quién es o quién lo dijo. Antes de responder, comprueba que el dato "
    "pertenece a la persona o cosa por la que se pregunta; si en la memoria pertenece a otra (otra persona, otro "
    "familiar, otro objeto), dilo y corrige la premisa de la pregunta en vez de atribuírselo. Si un recuerdo usa "
    "una expresión de tiempo relativa (ayer, la semana pasada, el mes que viene), calcula la fecha a partir de la "
    "fecha de ese recuerdo, no de la actual. Fecha actual: {now}."
)

TRUNCATED_NOTE = "\n\n[⚠️ Respuesta truncada por el límite de tokens (STM_ANSWER_MAX_TOKENS).]"

# [PARCHE1] Aviso para recuerdos de la banda tentativa (similitud dudosa): solo se usan si responden de verdad.
LTM_CAUTION = (
    "Los siguientes recuerdos se recuperaron por similitud y PUEDEN NO SER RELEVANTES para la pregunta. "
    "Úsalos solo si responden de verdad y directamente a lo que se pregunta; si no responden, di claramente "
    "que no tienes ese dato en memoria y no lo inventes ni lo deduzcas a partir de datos parecidos."
)


def split_chunks(content: str, max_chars: int = 2200) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()] or [content.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paragraphs:
        while len(p) > max_chars:
            cut = p.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            piece, p = p[:cut].strip(), p[cut:].strip()
            if cur:
                chunks.append(cur)
                cur = ""
            if piece:
                chunks.append(piece)
        if not p:
            continue
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


class Agent:
    def __init__(self, stm: ShortTermMemory | None = None, ltm_tree: MemoryTree | None = None,
                 llm: LLM | None = None, ltm: Any | None = None) -> None:
        self.llm = llm or LLM(base_url=C.LLM_STM_URL, model=C.REMOTE_MODEL_ALIAS)
        self.logs: list[str] = []
        self._log_lock = threading.Lock()
        self._chat_lock = threading.Lock()

        self.cartridge_mgr = CartridgeManager(self)
        self.ltm_trees: dict[str, Any] = {}
        primary_id = self.cartridge_mgr.get_primary_id()

        if ltm or ltm_tree:
            injected = ltm or ltm_tree
            self.ltm = injected
            self.ltm_trees[primary_id] = injected
        else:
            for cid in self.cartridge_mgr.get_mounted_ids():
                try:
                    cpath = self.cartridge_mgr.get_cartridge_path(cid)
                    self.ltm_trees[cid] = MemoryTree(cpath, MemLLM())
                except Exception as exc:
                    self.log(f"⚠️ Error cargando cartucho {cid}: {exc}")

            if primary_id not in self.ltm_trees:
                self.ltm_trees[primary_id] = MemoryTree(self.cartridge_mgr.get_primary_path(), MemLLM())
            self.ltm = self.ltm_trees[primary_id]

        active_path = self.cartridge_mgr.get_primary_path()
        self.stm = stm or ShortTermMemory(self.llm, path=active_path, on_event=self.log)

        self._backfill_episodes()
        self.ltm_lock = threading.Lock()
        self._ltm_pause_event = threading.Event()
        self._ltm_pause_event.set()
        self._pause_lock = threading.Lock()
        self._pause_depth = 0
        self._stop_event = threading.Event()

        self.server_ready = False
        self.state: dict[str, Any] = {
            "phase": "IDLE",
            "steps": {},
            "ltm_task": "IDLE",
            "ltm_detail": "Bucle en reposo",
            "ltm_elapsed": 0.0,
        }
        self.history: list[dict[str, Any]] = []
        self._backup_bucket = int(getattr(self.ltm, "tick", 0)) // max(1, int(getattr(C, "MEM_BACKUP_EVERY", 1) or 1))
        self._backup_rev = getattr(self.ltm, "rev", 0)

    def log(self, msg: str) -> None:
        line = time.strftime("[%H:%M:%S] ") + msg
        with self._log_lock:
            self.logs.append(line)
            del self.logs[:-120]
        print(line, flush=True)

    def _set(self, phase: str, **steps: str) -> None:
        self.state["phase"] = phase
        self.state["steps"].update(steps)

    def _pause_ltm(self) -> None:
        with self._pause_lock:
            self._pause_depth += 1
            self._ltm_pause_event.clear()

    def _resume_ltm(self) -> None:
        with self._pause_lock:
            self._pause_depth = max(0, self._pause_depth - 1)
            if self._pause_depth == 0:
                self._ltm_pause_event.set()

    def start_background_tasks(self) -> None:
        self.stm.start_background()
        if hasattr(self.ltm, "maintenance_step"):
            threading.Thread(target=self._ltm_maintenance_loop, name="ltm-maintenance", daemon=True).start()

    def _ltm_maintenance_loop(self) -> None:
        idle = 0
        base_sleep = max(0.5, C.MEM_IDLE_SLEEP_S)
        max_sleep = max(base_sleep, C.MEM_IDLE_MAX_SLEEP_S)
        while not self._stop_event.is_set():
            self._ltm_pause_event.wait(timeout=1.0)
            if not self._ltm_pause_event.is_set() or self._stop_event.is_set():
                continue
            try:
                with self.ltm_lock:
                    trees = [t for t in list(self.ltm_trees.values()) if hasattr(t, "maintenance_step") and hasattr(t, "conn") and t.conn] or [self.ltm]
                    if not trees:
                        time.sleep(0.2)
                        continue
                    self._mnt_turn = getattr(self, "_mnt_turn", 0) + 1
                    tree = self.ltm if (self._mnt_turn % 2 == 1 or len(trees) == 1) else trees[(self._mnt_turn // 2) % len(trees)]
                    if not hasattr(tree, "conn") or not tree.conn:
                        continue
                    tick = tree.advance(1)
                    task_name = tree.TASKS[(tree.step - 1) % len(tree.TASKS)]
                    self.state["ltm_task"] = task_name
                    self.state["ltm_detail"] = f"Paso {task_name} en tick #{tick:06d}..."
                    started = time.perf_counter()
                    name, report, worked = tree.maintenance_step()
                    elapsed = time.perf_counter() - started

                if tree is self.ltm:
                    self._maybe_backup(tick)
                self.state["ltm_elapsed"] = round(elapsed, 2)
                self.state["ltm_detail"] = f"[{name.upper()}] {report} ({elapsed:.1f}s)"
                if worked:
                    idle = 0
                    self.log(f"[LTM-{name}] {report} ({elapsed:.1f}s)")
                else:
                    idle += 1

                delay = min(max_sleep, base_sleep * 2 ** min(idle // 8, 6))
                for _ in range(int(delay * 10)):
                    if not self._ltm_pause_event.is_set() or self._stop_event.is_set():
                        break
                    time.sleep(0.1)
            except Exception as exc:
                self.log(f"[LTM-Error] {exc}")
                self._stop_event.wait(3)

    def _maybe_backup(self, tick: int, final: bool = False) -> None:
        every = int(getattr(C, "MEM_BACKUP_EVERY", 0) or 0)
        if not every or not hasattr(self.ltm, "conn"):
            return
        bucket = tick // every
        rev = getattr(self.ltm, "rev", 0)
        if rev == self._backup_rev or (not final and bucket <= self._backup_bucket):
            return
        try:
            from bucle import _backup
            _backup(tick, self.ltm)
            self._backup_bucket, self._backup_rev = bucket, rev
            self.log(f"[Backup] copia de memoria.db en body/backups (tick {tick})")
        except Exception as exc:
            self.log(f"[Backup] fallo: {exc}")

    def _query_ltm_in_process(self, question: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        self._pause_ltm()
        prev_task = self.state["ltm_task"]
        prev_detail = self.state["ltm_detail"]
        mounted_count = len(self.ltm_trees)
        self.state["ltm_task"] = "USER_PRIORITY"
        self.state["ltm_detail"] = f"Búsqueda en {mounted_count} cartucho(s) para: «{question[:35]}...»"

        def _query_single_tree(tree_obj: Any, q: str) -> dict[str, Any]:
            if hasattr(tree_obj, "recall"):
                return tree_obj.recall(q, tree_obj is self.ltm, synthesize=not bool(getattr(C, 'LTM_DIRECT_RETRIEVAL', False)))
            elif hasattr(tree_obj, "ask"):
                return tree_obj.ask(q)
            return {"found": False, "answer": ""}

        try:
            results = {}
            with ThreadPoolExecutor(max_workers=max(1, mounted_count)) as ex:
                futures = {
                    cid: ex.submit(_query_single_tree, tree, question.strip())
                    for cid, tree in list(self.ltm_trees.items())
                }
                for cid, f in futures.items():
                    try:
                        results[cid] = f.result()
                    except Exception as e:
                        results[cid] = {"found": False, "error": str(e)}

            # [PARCHE1] Separar recuerdos firmes de los tentativos (por cartucho) para avisar al LLM.
            firm_items: list[str] = []
            tent_items: list[str] = []
            all_sources = []
            all_cells = []
            all_methods = set()

            cart_names = {c["id"]: c["name"] for c in self.cartridge_mgr.list_cartridges()}

            for cid, res in results.items():
                if res.get("found") and res.get("answer"):
                    cname = cart_names.get(cid, cid)
                    ans = str(res.get("answer", "")).strip()
                    block = "📌 [Cartucho: " + str(cname) + "]\n" + ans
                    (tent_items if res.get("tentative") else firm_items).append(block)
                    all_sources.extend(res.get("sources", []))
                    all_cells.extend(res.get("cells", []))
                    all_methods.update(res.get("methods", []))

            has_found = bool(firm_items or tent_items)

            return {
                "found": has_found,
                "tentative": bool(tent_items) and not firm_items,
                "answer": "\n\n".join(firm_items),
                "tentative_answer": "\n\n".join(tent_items),
                "sources": sorted(set(all_sources)),
                "cells": all_cells,
                "methods": sorted(all_methods),
                "ms": round((time.perf_counter() - t0) * 1000),
                "cartridges_queried": list(self.ltm_trees.keys())
            }
        except Exception as exc:
            return {
                "found": False, "answer": "", "sources": [], "cells": [], "methods": [],
                "ms": round((time.perf_counter() - t0) * 1000), "error": str(exc)
            }
        finally:
            self.state["ltm_task"] = prev_task
            self.state["ltm_detail"] = prev_detail
            self._resume_ltm()

    def _build_system(self, stm: dict[str, Any], ltm: dict[str, Any],
                      epi: list[dict[str, Any]] | None = None) -> str:
        parts = [BASE_PROMPT.format(now=time.strftime("%d/%m/%Y %H:%M"))]
        if stm.get("summaries"):
            parts.append("\n## Resumen de la conversación previa (memoria corta)")
            for s in stm["summaries"]:
                parts.append(f"• {s['text']}")
        if stm.get("hits"):
            parts.append("\n## Mensajes antiguos relevantes de esta charla (memoria corta)")
            for h in stm["hits"]:
                parts.append(f"[{fmt_ts(h['ts'])} · {role_label(h)}] {h['text'][:500]}")
        # [PARCHE1] Recuerdos firmes y tentativos en bloques distintos; los tentativos llevan aviso.
        firm = str(ltm.get("answer") or "").strip() if ltm.get("found") else ""
        tent = str(ltm.get("tentative_answer") or "").strip() if ltm.get("found") else ""
        if firm and ltm.get("tentative") and "tentative_answer" not in ltm:
            firm, tent = "", firm          # resultado de un único árbol (formato antiguo)
        if firm:
            parts.append("\n## Conocimiento de Memoria a Largo Plazo (Cartuchos Conectados)")
            parts.append(firm)
        if tent:
            parts.append("\n## Recuerdos recuperados (posible relevancia)")
            parts.append(LTM_CAUTION)
            parts.append(tent)
        if (firm or tent) and ltm.get("sources"):
            parts.append("📚 Fuentes acreditadas: " + "; ".join(ltm["sources"]))
        if epi:
            seen = {str(e.get("text", "")).strip() for e in stm.get("recent", [])}
            seen |= {str(h.get("text", "")).strip() for h in stm.get("hits", [])}
            fresh = [e for e in epi if str(e["text"]).strip() not in seen][:5]
            if fresh:
                parts.append("\n## Recuerdos literales de conversaciones pasadas (memoria episódica)")
                parts.append("Estos episodios son transcripciones fechadas de lo que se dijo entonces, no necesariamente el estado actual. Para preguntas sobre el presente, prioriza los hechos de memoria a largo plazo cuya vigencia esté activa; usa episodios como contexto histórico y no como prueba de que un hecho siga vigente.")
                parts.append("Cada episodio indica quién habló: lo que alguien cuenta en primera persona («yo», «mi») "
                             "es de quien habla, no de su interlocutor.")  # [PARCHE12]
                for e in fresh:
                    parts.append(f"[{fmt_ts(e['ts'])} · {role_label(e)}] {e['text'][:500]}")
        return "\n".join(parts)

    def _episode(self, role: str, text: str) -> None:
        fn = getattr(self.ltm, "log_episode", None)
        if fn:
            try:
                fn(role, text)
            except Exception as exc:
                self.log(f"[Episódica] {exc}")

    def _episodes_for(self, text: str) -> list[dict[str, Any]]:
        fn = getattr(self.ltm, "search_episodes", None)
        if not fn:
            return []
        try:
            return fn(text, getattr(C, "EPI_LOOKUP_K", 6)) or []
        except Exception as exc:
            self.log(f"[Episódica] {exc}")
            return []

    def _backfill_episodes(self) -> None:
        fn = getattr(self.ltm, "backfill_episodes", None)
        if not fn:
            return
        try:
            n = fn(getattr(self.stm, "dir", C.STM_DIR))
            if n:
                self.log(f"[Episódica] {n} mensajes antiguos importados a la memoria episódica.")
        except Exception as exc:
            self.log(f"[Episódica] importación fallida: {exc}")

    def _ingest(self, text: str, origin: str) -> None:
        if hasattr(self.ltm, "ingest_async"):
            try:
                self.ltm.ingest_async(text, origin)
            except Exception as exc:
                self.log(f"[LTM-Ingesta] {exc}")

    def chat_stream(self, text: str):
        text = (text or "").strip()
        if not text:
            yield {"event": "error", "error": "Mensaje vacío"}
            return
        finished = False
        with self._chat_lock:
            try:
                t0 = time.perf_counter()
                self._set("RECEPCION", store="run")
                epi_hits = self._episodes_for(text)
                self.stm.add("user", text)
                self._episode("user", text)
                self._ingest(text, "mensaje del usuario")

                self._set("CONSULTA", stm_lookup="run", ltm_query="run")
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_stm = ex.submit(self.stm.recall, text)
                    f_ltm = ex.submit(self._query_ltm_in_process, text)
                    stm_ctx = f_stm.result()
                    ltm_res = f_ltm.result()

                self._set("CONSULTA", stm_lookup="done", ltm_query="done")
                self._set("RESPUESTA", answer="run")

                messages = [{"role": "system", "content": self._build_system(stm_ctx, ltm_res, epi_hits)}]
                messages += [{"role": e["role"], "content": e["text"]} for e in stm_ctx["recent"]]
                if messages[-1]["role"] != "user":
                    messages.append({"role": "user", "content": text})

                stm_info = {"hits": len(stm_ctx.get("hits", [])),
                            "summaries": len(stm_ctx.get("summaries", [])), "ms": stm_ctx.get("ms", 0)}
                yield {"event": "context", "stm": stm_info, "ltm": ltm_res}

                t1 = time.perf_counter()
                parts: list[str] = []
                try:
                    if hasattr(self.llm, "chat_stream"):
                        for delta in self.llm.chat_stream(messages):
                            parts.append(delta)
                            yield {"event": "token", "text": delta}
                    else:
                        whole = self.llm.chat(messages)
                        parts.append(whole)
                        yield {"event": "token", "text": whole}
                except Exception as exc:
                    self._set("IDLE", answer="error")
                    finished = True
                    yield {"event": "error", "error": f"Error LLM: {exc}"}
                    return

                answer = "".join(parts).strip()
                llm_ms = round((time.perf_counter() - t1) * 1000)
                truncated = getattr(self.llm, "last_finish_reason", None) == "length"
                self._set("ALMACENAR", answer="done", store="run")
                self.stm.add("assistant", answer)
                self._episode("assistant", answer)
                self._ingest(answer, "respuesta del agente")
                if truncated:
                    self.log("⚠️ Respuesta truncada por el límite de tokens; sube STM_ANSWER_MAX_TOKENS.")

                self._set("IDLE", store="done")
                res_obj = {
                    "answer": answer + (TRUNCATED_NOTE if truncated else ""), "error": None,
                    "stm": stm_info, "ltm": ltm_res,
                    "llm_ms": llm_ms, "total_ms": round((time.perf_counter() - t0) * 1000),
                    "tokens": self.stm.total_tokens(),
                }
                self.history.append({"ts": time.time(), "q": text[:100], "res": res_obj})
                del self.history[:-30]
                finished = True
                yield {"event": "done", "result": res_obj}
            finally:
                if not finished:
                    self._set("IDLE", answer="error")

    def chat(self, text: str) -> dict[str, Any]:
        for ev in self.chat_stream(text):
            if ev["event"] == "done":
                return ev["result"]
            if ev["event"] == "error":
                return {"answer": "", "error": ev["error"]}
        return {"answer": "", "error": "Sin respuesta"}

    def ingest_stream(self, content: str, source: str = ""):
        content = (content or "").strip()
        if not content:
            yield {"event": "error", "error": "Texto vacío"}
            return
        self._pause_ltm()
        prev_task = self.state["ltm_task"]
        self.state["ltm_task"] = "USER_PRIORITY"
        try:
            chunks = split_chunks(content)
            explicit = source.strip()
            effective_src = explicit or f"Ingesta manual el {time.strftime('%d/%m/%Y %H:%M')}"
            total = 0
            items: list[dict[str, Any]] = []
            for idx, chunk in enumerate(chunks):
                self.state["ltm_detail"] = f"Ingesta: extrayendo conceptos del bloque {idx + 1}/{len(chunks)}..."
                facts = None
                if hasattr(self.ltm, "extract_facts"):
                    detected, facts = self.ltm.extract_facts(chunk, first=(idx == 0 and not explicit))
                    if idx == 0 and not explicit and detected:
                        effective_src = detected
                if facts is None:
                    facts = [{"texto": chunk[:1200], "importancia": 7, "tipo": "dato"}]
                for f in facts:
                    imp = max(9.0, f["importancia"]) if f["tipo"] == "instruccion" else f["importancia"]
                    res = self.ltm.store(f["texto"], imp, f["tipo"], meta={"fuente": effective_src})
                    total += 1
                    items.append({"path": res.get("path"), "duplicate": res.get("duplicate", False)})
                yield {"event": "progress", "block": idx + 1, "total": len(chunks), "facts": total}

            self.log(f"[LTM-Ingesta] {total} hechos registrados bajo fuente: «{effective_src}»")
            yield {"event": "done", "result": {"success": True, "facts": total, "source": effective_src,
                                               "blocks": len(chunks), "items": items[:200]}}
        except Exception as exc:
            yield {"event": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.state["ltm_task"] = prev_task
            self.state["ltm_detail"] = "Ingesta completada"
            self._resume_ltm()

    def ingest_long_text(self, content: str, source: str = "") -> dict[str, Any]:
        for ev in self.ingest_stream(content, source):
            if ev["event"] == "done":
                return ev["result"]
            if ev["event"] == "error":
                return {"success": False, "error": ev["error"], "facts": 0}
        return {"success": False, "error": "Sin resultado", "facts": 0}

    def mount_cartridge(self, cartridge_id: str) -> dict[str, Any]:
        with self._chat_lock, self.ltm_lock:
            self._pause_ltm()
            try:
                self.log(f"🔌 Conectando cartucho auxiliar: «{cartridge_id}»...")
                self.cartridge_mgr.mount(cartridge_id)
                if cartridge_id not in self.ltm_trees:
                    p = self.cartridge_mgr.get_cartridge_path(cartridge_id)
                    self.ltm_trees[cartridge_id] = MemoryTree(p, MemLLM())
                self.log(f"✓ Cartucho «{cartridge_id}» conectado. Total activos: {len(self.ltm_trees)}")
                return {"ok": True, "mounted": list(self.ltm_trees.keys())}
            except Exception as e:
                self.log(f"❌ Error conectando cartucho {cartridge_id}: {e}")
                return {"ok": False, "error": str(e)}
            finally:
                self._resume_ltm()

    def unmount_cartridge(self, cartridge_id: str) -> dict[str, Any]:
        with self._chat_lock, self.ltm_lock:
            self._pause_ltm()
            try:
                if cartridge_id == self.cartridge_mgr.get_primary_id():
                    return {"ok": False, "error": "No puedes desconectar el cartucho primario de escritura. Selecciona otro como primario primero."}

                self.log(f"🔌 Desconectando cartucho: «{cartridge_id}»...")
                self.cartridge_mgr.unmount(cartridge_id)
                if cartridge_id in self.ltm_trees:
                    tree = self.ltm_trees.pop(cartridge_id)
                    if hasattr(tree, "close"):
                        tree.close()
                self.log(f"✓ Cartucho «{cartridge_id}» desconectado.")
                return {"ok": True, "mounted": list(self.ltm_trees.keys())}
            except Exception as e:
                self.log(f"❌ Error desconectando cartucho {cartridge_id}: {e}")
                return {"ok": False, "error": str(e)}
            finally:
                self._resume_ltm()

    def set_primary_cartridge(self, cartridge_id: str) -> dict[str, Any]:
        with self._chat_lock, self.ltm_lock:
            self._pause_ltm()
            try:
                self.log(f"✍️ Estableciendo cartucho primario de escritura: «{cartridge_id}»...")
                self.cartridge_mgr.set_primary(cartridge_id)
                if cartridge_id not in self.ltm_trees:
                    p = self.cartridge_mgr.get_cartridge_path(cartridge_id)
                    self.ltm_trees[cartridge_id] = MemoryTree(p, MemLLM())

                self.ltm = self.ltm_trees[cartridge_id]
                self.stm.stop()
                self.stm = ShortTermMemory(self.llm, path=self.cartridge_mgr.get_primary_path(), on_event=self.log)
                self.stm.start_background()
                self.cartridge_mgr.sync_to_legacy(cartridge_id)
                self.log(f"✓ Cartucho «{cartridge_id}» es ahora el primario de escritura.")
                return {"ok": True, "primary": cartridge_id}
            except Exception as e:
                self.log(f"❌ Error cambiando cartucho primario: {e}")
                return {"ok": False, "error": str(e)}
            finally:
                self._resume_ltm()

    def switch_cartridge(self, cartridge_id: str) -> dict[str, Any]:
        return self.set_primary_cartridge(cartridge_id)

    def reset_cartridge(self, cartridge_id: str | None = None) -> dict[str, Any]:
        with self._chat_lock, self.ltm_lock:
            self._pause_ltm()
            try:
                cid = cartridge_id or self.cartridge_mgr.get_primary_id()
                self.log(f"⚠️ Vaciando contenido del cartucho: «{cid}»...")
                res = self.cartridge_mgr.reset_cartridge(cid)
                if res.get("ok"):
                    self.log(f"✓ Contenido del cartucho «{cid}» vaciado.")
                return res
            except Exception as e:
                self.log(f"❌ Error vaciando cartucho: {e}")
                return {"ok": False, "error": str(e)}
            finally:
                self._resume_ltm()

    def reset_active_cartridge(self) -> dict[str, Any]:
        return self.reset_cartridge()

    def get_graph_data(self, cartridge_id: str | None = None) -> dict[str, Any]:
        """Obtiene la topología de nodos y sinapsis de un cartucho o de todos los conectados."""
        if cartridge_id and cartridge_id != "all":
            tree = self.ltm_trees.get(cartridge_id)
            if tree and hasattr(tree, "get_graph_data"):
                data = tree.get_graph_data()
                for n in data.get("nodes", []):
                    n["cartridge"] = cartridge_id
                return data
            return {"nodes": [], "synapses": []}

        all_nodes = []
        all_synapses = []
        cart_names = {}
        if hasattr(self, "cartridge_mgr"):
            cart_names = {c["id"]: c["name"] for c in self.cartridge_mgr.list_cartridges()}

        single = len(self.ltm_trees) <= 1
        for cid, tree in list(self.ltm_trees.items()):
            if not hasattr(tree, "get_graph_data"):
                continue
            data = tree.get_graph_data()
            cname = cart_names.get(cid, cid.capitalize())

            for n in data.get("nodes", []):
                n_copy = dict(n)
                n_copy["orig_id"] = n["id"]
                n_copy["id"] = n["id"] if single else f"{cid}_{n['id']}"
                if n.get("parent_id") is not None and not single:
                    n_copy["parent_id"] = f"{cid}_{n['parent_id']}"
                n_copy["cartridge"] = cid
                n_copy["cartridge_name"] = cname
                all_nodes.append(n_copy)

            for s in data.get("synapses", []):
                s_copy = dict(s)
                s_copy["source_id"] = s["source_id"] if single else f"{cid}_{s['source_id']}"
                s_copy["target_id"] = s["target_id"] if single else f"{cid}_{s['target_id']}"
                s_copy["cartridge"] = cid
                all_synapses.append(s_copy)

        return {"nodes": all_nodes, "synapses": all_synapses}

    def status(self) -> dict[str, Any]:
        with self._log_lock:
            logs = list(self.logs)
        counts = self.ltm.counts() if hasattr(self.ltm, "counts") else {}
        ltm_summary = self.ltm.summary() if hasattr(self.ltm, "summary") else ""
        return {
            "phase": self.state["phase"],
            "steps": dict(self.state["steps"]),
            "server_ready": self.server_ready,
            "stm": self.stm.stats_snapshot(),
            "ltm": {
                "tick": getattr(self.ltm, "tick", 0),
                "rev": getattr(self.ltm, "rev", 0),
                "alive": counts.get("alive", 0),
                "archived": counts.get("archived", 0),
                "nodes": counts.get("nodes", 0),
                "dirty": counts.get("dirty", 0),
                "ingest_queue": counts.get("ingest_queue", 0),
                "episodes": counts.get("episodes", 0),
                "vectors": counts.get("vectors", 0),
                "stats": dict(getattr(self.ltm, "stats", {})),
                "summary": ltm_summary,
                "current_task": self.state["ltm_task"],
                "detail": self.state["ltm_detail"],
                "elapsed": self.state.get("ltm_elapsed", 0.0),
            },
            "cartridge": self.cartridge_mgr.get_active_info() if hasattr(self, "cartridge_mgr") else {},
            "mounted_cartridges": self.cartridge_mgr.get_mounted_ids() if hasattr(self, "cartridge_mgr") else [],
            "logs": logs,
            "system_info": {
                "python": platform.python_version(),
                "gpu0_port": C.SERVER_PORT_GPU0,
                "gpu1_port": C.SERVER_PORT_GPU1,
                "model": C.REMOTE_MODEL_ALIAS,
            },
        }

    def stop(self) -> None:
        self._stop_event.set()
        self._maybe_backup(int(getattr(self.ltm, "tick", 0)), final=True)
        self._ltm_pause_event.set()
        self.stm.stop()
        if hasattr(self, "cartridge_mgr"):
            try:
                self.cartridge_mgr.sync_to_legacy(self.cartridge_mgr.get_primary_id())
            except Exception:
                pass
        for tree in self.ltm_trees.values():
            if hasattr(tree, "shutdown"):
                try:
                    tree.shutdown()
                except Exception:
                    pass
