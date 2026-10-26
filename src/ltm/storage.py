# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import re
import secrets
import sqlite3
import time
import unicodedata
from typing import Any

import config as _cfg
from .common import N, LEVELS, SYSTEM, clamp, slugify, clean_path_specs


class StorageMixin:
    """Lógica de persistencia de hechos atómicos, ontología jerárquica y reconciliación temporal."""

    def _path_str(self, node_id: int) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                WITH RECURSIVE path_cte AS (
                    SELECT id, parent_id, name, level, name as full_path
                    FROM nodes WHERE id = ?
                    UNION ALL
                    SELECT n.id, n.parent_id, n.name, n.level, n.name || ' › ' || p.full_path
                    FROM nodes n JOIN path_cte p ON n.id = p.parent_id
                )
                SELECT full_path FROM path_cte WHERE parent_id IS NULL;
            """, (node_id,))
            r = cur.fetchone()
            return r[0] if r else ""

    def _kids_sorted(self, parent_id: int | None) -> list[sqlite3.Row]:
        with self.lock:
            cur = self.conn.cursor()
            if parent_id is None:
                cur.execute("SELECT * FROM nodes WHERE parent_id IS NULL ORDER BY importance DESC, name ASC")
            else:
                cur.execute("SELECT * FROM nodes WHERE parent_id = ? ORDER BY importance DESC, name ASC", (parent_id,))
            return cur.fetchall()

    def _mark_dirty(self, node_id: int) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                WITH RECURSIVE ancestors AS (
                    SELECT id, parent_id FROM nodes WHERE id = ?
                    UNION ALL
                    SELECT n.id, n.parent_id FROM nodes n JOIN ancestors a ON n.id = a.parent_id
                )
                UPDATE nodes SET dirty = 1 WHERE id IN (SELECT id FROM ancestors);
            """, (node_id,))
            self.conn.commit()

    def _make_node(self, parent_id: int | None, spec: dict[str, Any], level: int) -> int:
        name = str(spec.get("name") or "general").strip()[:60] or "general"
        desc = str(spec.get("description") or "").strip()[:200]
        slug = slugify(name)
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id, description FROM nodes WHERE (parent_id IS ? OR parent_id = ?) AND slug = ?",
                (parent_id, parent_id, slug)
            )
            row = cur.fetchone()
            if row:
                if not row["description"] and desc:
                    cur.execute("UPDATE nodes SET description = ? WHERE id = ?", (desc, row["id"]))
                    self.conn.commit()
                return row["id"]
            cur.execute("""
                INSERT INTO nodes (parent_id, level, slug, name, description, dirty, created_tick)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """, (parent_id, level, slug, name, desc, self.tick))
            self.conn.commit()
            self._bump()
            return cur.lastrowid

    def eff(self, cell: dict[str, Any] | sqlite3.Row) -> float:
        imp = float(cell["importance"])
        age = max(0, self.tick - int(cell["updated_tick"]))
        return imp * math.exp(-age / (_cfg.MEM_TAU * max(imp, 1.0) ** 3))

    def _reinforce(self, cell_id: str) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE cells 
                SET importance = MIN(10.0, importance + MAX(0.15, (10.0 - importance) / 5.0)),
                    updated_tick = ?,
                    last_access_tick = ?,
                    access_count = access_count + 1
                WHERE id = ?
            """, (self.tick, self.tick, cell_id))
            self.conn.commit()

    def _reinforce_synapse(self, node_a: int, node_b: int, delta: float = 0.5) -> None:
        if node_a == node_b or not getattr(_cfg, "SYN_ENABLED", True):
            return
        with self.lock:
            cur = self.conn.cursor()
            for src, dst in ((node_a, node_b), (node_b, node_a)):
                cur.execute("""
                    INSERT INTO synapses (source_id, target_id, weight, coactivations, last_tick, created_tick)
                    VALUES (?, ?, 1.0 + ?, 1, ?, ?)
                    ON CONFLICT(source_id, target_id) DO UPDATE SET
                        weight = MIN(10.0, weight + MAX(0.1, (10.0 - weight) / 8.0)),
                        coactivations = coactivations + 1,
                        last_tick = ?
                """, (src, dst, delta, self.tick, self.tick, self.tick))
            self.conn.commit()

    @staticmethod
    def _looks_like_temporal_update(old_text: str, new_text: str) -> bool:
        def fold(value: str) -> str:
            value = unicodedata.normalize("NFKD", str(value).lower())
            return "".join(ch for ch in value if not unicodedata.combining(ch))
        old, new = fold(old_text), fold(new_text)
        cues = (
            "mudado", "se ha mudado", "se mudo", "me he mudado", "me mude", "mudanza", "traslado", "trasladado", "me traslade", "he cambiado", "cambio de domicilio",
            "ahora vivo", "ahora trabajo", "ahora estudio", "ya no vivo", "ya no trabajo", "ya no estudio",
            "a partir de ahora", "desde ahora", "actualmente", "mi nuevo empleo", "nuevo domicilio",
            "cambie de", "deje de", "he dejado de", "empece a", "renuncie", "me despidieron",
            "moved to", "i moved", "changed my", "no longer", "now i live", "now i work",
            "started working", "left my job", "my new address", "currently live",
            "vive ahora", "vivo ahora", "trabaja ahora", "trabajo ahora", "ahora vive", "ahora trabaja",
            "fue despedido", "fue despedida", "despedido de", "ya no ", "dejo de", "termino la carrera",
            "se ha graduado", "se graduo", "se caso", "se divorcio", "se separo", "se jubilo", "se ha jubilado",
            "cambio de trabajo", "cambio de ciudad", "nuevo trabajo",
        )
        if any(cue in new for cue in cues):
            return True
        patterns = (
            r"\b(?:vivo|vive|vives|vivimos|viven|resido|reside|resides|residimos|residen|habito|habita|habitan)\s+en\s+(.+)",
            r"\b(?:trabajo|trabaja|trabajas|trabajamos|trabajan|estudio|estudia|estudias|estudiamos|estudian)\s+en\s+(.+)",
            r"\bmi\s+(?:domicilio|direccion|ciudad|empresa|puesto|cargo|pareja|perro|gato|mascota)\s+(?:es|se llama)\s+(.+)",
            r"\b(?:prefiero|mi preferencia es)\s+(.+)",
            r"\bmi edad es\s+(.+)",
        )
        extra = (
            r"\b(?:vivo|vive|resido|reside)\s+(?:ahora\s+|actualmente\s+)?en\s+(.+)",
            r"\b(?:trabajo|trabaja)\s+(?:ahora\s+|actualmente\s+)?(?:en|para)\s+(.+)",
            r"\b(?:estudio|estudia)\s+(?:ahora\s+|actualmente\s+)?(?:en\s+)?(.+)",
            r"\b(?:prefiero|prefiere)\s+(.+)",
            r"\b(?:tengo|tiene)\s+(\d+)\s+anos",
        )
        for pattern in patterns + extra:
            before = re.search(pattern, old)
            after = re.search(pattern, new)
            if before and after:
                b = before.group(1).strip(" .,!;:")
                a = after.group(1).strip(" .,!;:")
                if b != a and not a.startswith(b) and not b.startswith(a):
                    return True
        return False

    def _detect_temporal_supersession(self, new_text: str, new_valid_from: str,
                                      near: list[tuple[dict[str, Any], float]] | None = None) -> str | None:
        if os.getenv("HMEM_TEMPORAL_RECONCILE", "1").strip().lower() in {"0", "false", "no", "off"}:
            return None
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        near_min = max(0.35, float(getattr(_cfg, "EMB_MIN_SIM", 0.45)) - 0.10)
        for cell, score in (near or []):
            cid = str(cell.get("id", ""))
            if (cid and cid not in seen and float(score) >= near_min
                    and not cell.get("archived") and not cell.get("valid_until")
                    and not cell.get("superseded_by")):
                candidates.append(cell)
                seen.add(cid)
        if not candidates:
            try:
                for cell in self._fts_search(self._fts_terms(new_text), limit=5):
                    cid = str(cell.get("id", ""))
                    if (cid and cid not in seen and not cell.get("archived")
                            and not cell.get("valid_until") and not cell.get("superseded_by")):
                        candidates.append(cell)
                        seen.add(cid)
            except Exception:
                candidates = []
        if not candidates and self._looks_like_temporal_update("", new_text):
            try:
                expanded = self._expand_query(new_text)
                terms = self._fts_terms(" ".join(expanded)) if expanded else []
                for cell in self._fts_search(terms, limit=8):
                    cid = str(cell.get("id", ""))
                    if (cid and cid not in seen and not cell.get("archived")
                            and not cell.get("valid_until") and not cell.get("superseded_by")):
                        candidates.append(cell)
                        seen.add(cid)
            except Exception:
                pass

        for old in candidates[:3]:
            old_from = str(old.get("valid_from") or old.get("ingest_timestamp") or "")
            if not self._looks_like_temporal_update(str(old.get("text", "")), new_text):
                continue
            if old_from and new_valid_from and new_valid_from < old_from:
                continue
            prompt = (
                "CLASIFICA LA RELACIÓN TEMPORAL ENTRE DOS HECHOS DE UNA MEMORIA PERSONAL.\n"
                f"Hecho previo (id={old.get('id')}): {old.get('text', '')}\n"
                f"Hecho nuevo (fecha de vigencia propuesta={new_valid_from}): {new_text}\n\n"
                "Devuelve SOLO JSON con relation, confidence y reason. relation debe ser una de: "
                "'supersedes', 'same', 'complementary', 'conflict' o 'unrelated'.\n"
                "Usa 'supersedes' ÚNICAMENTE si ambos hechos se refieren al mismo sujeto y atributo/estado, "
                "y el nuevo expresa un cambio temporal de estado (por ejemplo, mudanza de domicilio, "
                "nuevo empleo o cambio de preferencia explícito).\n"
                '{"relation":"supersedes|same|complementary|conflict|unrelated","confidence":0.0,"reason":"..."}'
            )
            try:
                obj = self.llm.ask_json(SYSTEM, prompt)
                if not isinstance(obj, dict):
                    continue
                relation = str(obj.get("relation", "")).strip().lower()
                try:
                    confidence = float(obj.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                if relation == "supersedes" and confidence >= 0.85:
                    return str(old["id"])
            except Exception:
                continue
        return None

    @staticmethod
    def _wants_history(question: str) -> bool:
        q = unicodedata.normalize("NFKD", str(question).lower())
        q = "".join(ch for ch in q if not unicodedata.combining(ch))
        explicit = ("antes", "anteriormente", "historial", "historico", "historica", "en el pasado",
                    "previamente", "solia", "cuando vivia", "donde vivia", "cuando trabajaba",
                    "donde trabajaba", "que habia dicho", "que dije", "que dijiste", "residencia anterior",
                    "domicilio anterior", "antes de mudarme", "version anterior")
        return any(term in q for term in explicit) or bool(re.search(r"\b(?:19|20)\d{2}\b", q))

    def store(self, text: str, importance: Any, kind: str = "dato", meta: dict[str, Any] | None = None) -> dict[str, Any]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return {"id": "", "path": "", "importance": 0.0, "duplicate": False, "empty": True}
        
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, node_id, importance FROM cells WHERE rtrim(lower(text), '.;!¡?¿ ') = rtrim(lower(?), '.;!¡?¿ ') AND archived = 0 AND valid_until IS NULL AND superseded_by IS NULL", (text,))
            exact = cur.fetchone()
            if exact:
                self._reinforce(exact["id"])
                return {
                    "id": exact["id"],
                    "path": self._path_str(exact["node_id"]),
                    "importance": min(10.0, exact["importance"] + 1.0),
                    "duplicate": True
                }

        imp = clamp(importance, 1, 10, 5)
        _meta = meta or {}
        _new_valid_from = str(_meta.get("valid_from") or _meta.get("ingest_timestamp") or (time.strftime("%Y-%m-%d %H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"))
        cur_node_id = None
        depth = 0
        _emb = getattr(self, "emb", None)
        _qvec = None
        _near: list[tuple[dict[str, Any], float]] = []
        if _emb is not None and getattr(_cfg, "EMB_PLACE", True):
            try:
                _qvec, _near = _emb.neighbors(text, k=5)
                _node, _shared = _emb.pick_placement(_near)
                if _node is not None:
                    cur_node_id = _node
                    depth = N if _shared is None else _shared
            except Exception:
                cur_node_id, depth = None, 0

        _supersedes_id = self._detect_temporal_supersession(text, _new_valid_from, _near)

        _adaptive = bool(getattr(_cfg, "MEM_ADAPTIVE_DEPTH", True))
        _max_depth = min(N, max(1, int(getattr(_cfg, "MEM_MAX_DEPTH", 5)))) if _adaptive else N
        while depth < _max_depth:
            kids = self._kids_sorted(cur_node_id)
            remaining = _max_depth - depth
            if _adaptive and cur_node_id is not None and not kids:
                break
            if kids:
                listing = "\n".join(f"- {k['slug']} ({k['name']}) — {k['description'] or 'sin descripción'}" for k in kids[:_cfg.MEM_LIST_LIMIT])
                if _adaptive:
                    here_opt = ('Si el dato es general para el nodo actual y no encaja en ninguno de sus hijos devuelve {"use":null,"here":true}.\n'
                                if cur_node_id is not None else "")
                    user = (
                        "Dato a memorizar: " + text + "\n"
                        "Profundidad actual: " + str(depth) + " (máximo " + str(_max_depth) + "). "
                        "La célula debe colgar del nodo MÁS ESPECÍFICO que haga falta, sin ramas de relleno.\n"
                        "Nodos existentes en este nivel:\n" + listing + "\n"
                        'Si alguno es el lugar adecuado devuelve {"use":"<slug exacto>","update_description":null}.\n'
                        + here_opt +
                        'Si ninguno encaja devuelve {"use":null,"path":[{"name":"...","description":"..."}]} con una rama NUEVA '
                        "de 1 a " + str(remaining) + " nodos (solo los necesarios; lo habitual es 1 o 2)."
                    )
                else:
                    user = (
                        "Dato a memorizar: " + text + "\n"
                        "Nivel actual: " + LEVELS[depth] + " (quedan " + str(remaining) + " niveles de nodos hasta la célula).\n"
                        "Nodos existentes en este nivel:\n" + listing + "\n"
                        'Si alguno es el lugar adecuado devuelve {"use":"<slug exacto>","update_description":null}.\n'
                        'Si ninguno encaja devuelve {"use":null,"path":[{"name":"...","description":"..."}]} creando una rama nueva.'
                    )
                try:
                    dec = self.llm.ask_json(SYSTEM, user)
                except Exception:
                    dec = {}
                if not isinstance(dec, dict):
                    dec = {}
                use_slug = dec.get("use")
                matched_kid = next((k for k in kids if k["slug"] == slugify(use_slug)), None) if use_slug else None
                if matched_kid:
                    cur_node_id = matched_kid["id"]
                    depth += 1
                    continue
                if _adaptive and cur_node_id is not None and dec.get("here"):
                    break
                path_specs = dec.get("path")
                if not isinstance(path_specs, list) or not path_specs:
                    if _adaptive and cur_node_id is not None:
                        break
                    path_specs = [{"name": "general", "description": ""}] * (1 if _adaptive else remaining)
            else:
                if _adaptive:
                    user = (
                        "Dato a memorizar: " + text + "\n"
                        "Crea la rama mínima necesaria para ubicarlo: de 1 a " + str(remaining) + " nodos, de lo general a lo "
                        "específico (lo habitual son 2 o 3; no añadas niveles de relleno ni nodos llamados 'general').\n"
                        'Devuelve JSON: {"path":[{"name":"...","description":"..."}]}'
                    )
                else:
                    user = (
                        "Dato a memorizar: " + text + "\n"
                        "Crea la rama completa de " + str(remaining) + " nodos desde '" + LEVELS[depth] + "' hasta '" + LEVELS[-1] + "'.\n"
                        'Devuelve JSON: {"path":[{"name":"...","description":"..."}]}'
                    )
                try:
                    obj = self.llm.ask_json(SYSTEM, user)
                    path_specs = obj.get("path") if isinstance(obj, dict) else []
                except Exception:
                    path_specs = []
                if not isinstance(path_specs, list) or not path_specs:
                    path_specs = [{"name": "general", "description": ""}] * (1 if _adaptive else remaining)

            path_specs = list(path_specs[:remaining])
            if _adaptive:
                path_specs = clean_path_specs(path_specs)
                if not path_specs:
                    if cur_node_id is not None:
                        break
                    path_specs = [{"name": "Sin clasificar", "description": ""}]
            if not _adaptive:
                while len(path_specs) < remaining:
                    path_specs.append({"name": "general", "description": ""})
            for s in path_specs:
                cur_node_id = self._make_node(cur_node_id, s if isinstance(s, dict) else {"name": str(s)}, depth)
                depth += 1
            break
        if cur_node_id is None:
            cur_node_id = self._make_node(None, {"name": "general", "description": ""}, 0)

        cid = f"{self.tick:07d}-{secrets.token_hex(6)}"
        with self.lock:
            cur = self.conn.cursor()
            fuente = meta.get("fuente") if meta else None
            cur.execute("SELECT id, node_id, importance FROM cells WHERE rtrim(lower(text), '.;!¡?¿ ') = rtrim(lower(?), '.;!¡?¿ ') AND archived = 0 AND valid_until IS NULL AND superseded_by IS NULL", (text,))
            _dup = cur.fetchone()
            if _dup:
                self._reinforce(_dup["id"])
                return {"id": _dup["id"], "path": self._path_str(_dup["node_id"]),
                        "importance": min(10.0, _dup["importance"] + 1.0), "duplicate": True}
            ingest_ts = meta.get("ingest_timestamp") if meta else time.strftime("%Y-%m-%d %H:%M:%S")
            cur.execute("""
                INSERT INTO cells (id, node_id, text, importance, kind, fuente, ingest_timestamp,
                                   created_tick, updated_tick, valid_from)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cid, cur_node_id, text, imp, kind, fuente, ingest_ts,
                  self.tick, self.tick, _new_valid_from))
            cur.execute("UPDATE nodes SET celulas = celulas + 1 WHERE id = ?", (cur_node_id,))
            if _supersedes_id:
                cur.execute("UPDATE cells SET valid_until = ?, superseded_by = ?, archived = 1 "
                            "WHERE id = ? AND valid_until IS NULL AND superseded_by IS NULL AND archived = 0",
                            (_new_valid_from, cid, _supersedes_id))
                if cur.rowcount:
                    cur.execute("UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells "
                                "WHERE node_id = nodes.id AND archived = 0) "
                                "WHERE id = (SELECT node_id FROM cells WHERE id = ?)", (_supersedes_id,))
                else:
                    _supersedes_id = None
            self.conn.commit()

        self._mark_dirty(cur_node_id)
        self._bump()
        if _emb is not None and _qvec is not None:
            try:
                _emb.put("cell", cid, _qvec)
            except Exception:
                pass
        return {"id": cid, "path": self._path_str(cur_node_id), "importance": imp,
                "duplicate": False, "superseded": _supersedes_id}

    def _add_directive(self, text: str, importance: float = 9.0) -> None:
        self.store(text, importance=max(9.0, importance), kind="instruccion")
