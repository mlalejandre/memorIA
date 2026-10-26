# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any

import config as _cfg
from .common import N, SYSTEM, clamp, slugify


class HomeostasisMixin:
    """Ciclo de sueño: auto-mantenimiento, poda, síntesis, autotest y balanceo."""

    def maintenance_step(self) -> tuple[str, str, bool]:
        name = self.TASKS[(self.step - 1) % len(self.TASKS)] if self.step > 0 else self.TASKS[0]
        calls0 = self.llm.calls
        worked = False
        try:
            report = getattr(self, "task_" + name)()
            worked = self.llm.calls != calls0 or "·" not in report
        except Exception as exc:
            report = f"error {type(exc).__name__}: {exc}"
        return name, report, worked

    def task_repair(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells "
                "WHERE cells.node_id = nodes.id AND cells.archived = 0)"
            )
            cur.execute("DELETE FROM synapses WHERE source_id NOT IN (SELECT id FROM nodes) OR target_id NOT IN (SELECT id FROM nodes)")
            cur.execute("UPDATE nodes SET parent_id = NULL WHERE parent_id IS NOT NULL AND parent_id NOT IN (SELECT id FROM nodes)")
            pruned = 0
            while True:
                cur.execute(
                    "DELETE FROM nodes WHERE pinned = 0 AND created_tick < ? "
                    "AND NOT EXISTS (SELECT 1 FROM cells WHERE cells.node_id = nodes.id) "
                    "AND NOT EXISTS (SELECT 1 FROM nodes ch WHERE ch.parent_id = nodes.id)",
                    (self.tick,),
                )
                if cur.rowcount <= 0:
                    break
                pruned += cur.rowcount
            cur.execute("SELECT COUNT(*) FROM cells")
            n_cells = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM cells_fts")
            n_fts = cur.fetchone()[0]
            rebuilt = False
            if n_cells != n_fts:
                cur.execute("DELETE FROM cells_fts")
                cur.execute("INSERT INTO cells_fts(cell_id, text) SELECT id, text FROM cells")
                rebuilt = True
            self.conn.commit()
        if pruned or rebuilt:
            self._bump()
            parts = []
            if pruned:
                parts.append(f"{pruned} nodo(s) vacío(s) podado(s)")
            if rebuilt:
                parts.append(f"índice FTS5 reconstruido ({n_fts} -> {n_cells})")
            return "; ".join(parts)
        return "· integridad B-Tree y FTS5 verificada"

    def task_decay(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE cells 
                SET archived = 1,
                    ruta_original = (SELECT name FROM nodes WHERE nodes.id = cells.node_id)
                WHERE archived = 0 
                  AND (? - updated_tick) >= ?
                  AND eff(importance, updated_tick, ?, ?) < ?
            """, (self.tick, _cfg.MEM_MIN_AGE, self.tick, _cfg.MEM_TAU, _cfg.MEM_ARCHIVE_BELOW))
            count = cur.rowcount
            
            tau_syn = float(getattr(_cfg, "SYN_TAU", 80.0))
            prune_below = float(getattr(_cfg, "SYN_PRUNE_BELOW", 0.3))
            cur.execute("""
                UPDATE synapses
                SET weight = weight * exp(-CAST(? - last_tick AS REAL) / ?)
                WHERE (? - last_tick) >= 10
            """, (self.tick, tau_syn, self.tick))
            cur.execute("UPDATE synapses SET last_tick = ? WHERE (? - last_tick) >= 10", (self.tick, self.tick))
            cur.execute("DELETE FROM synapses WHERE weight < ?", (prune_below,))
            syn_pruned = cur.rowcount
            
            if count > 0 or syn_pruned > 0:
                self.conn.commit()
                self._bump()
                parts = []
                if count > 0:
                    parts.append(f"{count} célula(s) archivadas por baja importancia efectiva")
                if syn_pruned > 0:
                    parts.append(f"{syn_pruned} sinapsis podadas")
                return "; ".join(parts)
        return "· células vivas estables, ninguna que olvidar"

    def task_describe(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, name, level FROM nodes WHERE dirty = 1 ORDER BY level DESC LIMIT ?", (_cfg.MEM_DESCRIBE_BATCH,))
            targets = [dict(r) for r in cur.fetchall()]

        if not targets:
            return "· descripciones al día"

        for n in targets:
            nid = n["id"]
            with self.lock:
                cur = self.conn.cursor()
                cur.execute("SELECT name, description FROM nodes WHERE parent_id = ? LIMIT 8", (nid,))
                items = [f"{r['name']}: {r['description']}" for r in cur.fetchall()]
                cur.execute("SELECT text FROM cells WHERE node_id = ? AND archived = 0 LIMIT 10", (nid,))
                items += [r["text"] for r in cur.fetchall()]

            if items:
                user = (
                    f"Nodo '{n['name']}' (profundidad {int(n['level']) + 1}).\n"
                    "Contenido:\n- " + "\n- ".join(items) + "\n"
                    "Escribe una descripción del ÁMBITO de este nodo en máx 20 palabras.\n"
                    'Devuelve: {"description": "..."}'
                )
                try:
                    obj = self.llm.ask_json(SYSTEM, user)
                    desc = str(obj.get("description", "")).strip()[:200]
                except Exception:
                    desc = ""
            else:
                desc = "Área general"

            with self.lock:
                cur = self.conn.cursor()
                cur.execute("UPDATE nodes SET description = ?, dirty = 0 WHERE id = ?", (desc, nid))
                self.conn.commit()

        return f"{len(targets)} descripciones sintetizadas"

    def task_dedupe(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT node_id FROM cells 
                WHERE archived = 0 
                GROUP BY node_id 
                HAVING COUNT(*) >= 2 
                ORDER BY RANDOM() LIMIT 1
            """)
            row = cur.fetchone()
            if not row:
                return "· sin tejidos con múltiples células para deduplicar"
            node_id = row[0]
            cur.execute("SELECT id, text, importance FROM cells WHERE node_id = ? AND archived = 0", (node_id,))
            cells = [dict(r) for r in cur.fetchall()]

        for i in range(len(cells)):
            w1 = set(re.findall(r"\w+", cells[i]["text"].lower()))
            for j in range(i + 1, len(cells)):
                w2 = set(re.findall(r"\w+", cells[j]["text"].lower()))
                inter = len(w1 & w2)
                union = len(w1 | w2)
                sim = inter / union if union > 0 else 0.0
                
                cos = self._cos_cells(cells[i]["id"], cells[j]["id"])
                if (sim >= 0.55 or cos >= float(getattr(_cfg, "EMB_DEDUPE_SIM", 0.88))
                        or cells[i]["text"].strip().lower() == cells[j]["text"].strip().lower()):
                    c1, c2 = cells[i], cells[j]
                    if c1["text"].strip().lower() == c2["text"].strip().lower():
                        merged_text = c1["text"]
                        new_imp = min(10.0, max(c1["importance"], c2["importance"]) + 0.5)
                    else:
                        prompt = (
                            f"Compara estos dos recuerdos del mismo tema:\n"
                            f"A: {c1['text']}\nB: {c2['text']}\n\n"
                            f"Si son redundantes o uno complementa al otro, unifícalos en una sola afirmación concisa. "
                            f'Devuelve JSON: {{"redundant": true, "merged": "...", "importance": 8}}.\n'
                            f'Si representan hechos distintos y deben conservarse ambos, devuelve: {{"redundant": false}}.'
                        )
                        try:
                            ans = self.llm.ask_json(SYSTEM, prompt)
                            if not ans.get("redundant"):
                                continue
                            merged_text = ans.get("merged", c1["text"])
                            new_imp = float(ans.get("importance", max(c1["importance"], c2["importance"])))
                        except Exception:
                            continue

                    with self.lock:
                        cur = self.conn.cursor()
                        cur.execute("UPDATE cells SET text = ?, importance = ?, updated_tick = ? WHERE id = ?",
                                    (merged_text, new_imp, self.tick, c1["id"]))
                        cur.execute("UPDATE cells SET archived = 1, updated_tick = ?, ruta_original = 'fusión con ' || ? WHERE id = ?",
                                    (self.tick, c1["id"], c2["id"]))
                        cur.execute("UPDATE nodes SET celulas = MAX(0, celulas - 1) WHERE id = ?", (node_id,))
                        self.conn.commit()
                    self._mark_dirty(node_id)
                    self._bump()
                    return f"fusión redundancia: [{c1['id']}] + [{c2['id']}] → «{merged_text[:35]}...»"

        return "· células revisadas, sin redundancias detectadas"

    def _mark_tested(self, cell_id: str) -> None:
        with self.lock:
            self.conn.execute("UPDATE cells SET tested_rev = ?, tested_tick = ? WHERE id = ?",
                              (self.rev, self.tick, cell_id))
            self.conn.commit()

    def _reindex_cell(self, cell_id: str) -> bool:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM cells_fts WHERE cell_id = ?", (cell_id,))
            if cur.fetchone()[0] > 0:
                return False
            cur.execute("INSERT INTO cells_fts(cell_id, text) SELECT id, text FROM cells WHERE id = ?", (cell_id,))
            self.conn.commit()
            return True

    def task_selftest(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM cells WHERE archived = 0 AND tested_rev < ? "
                        "ORDER BY tested_tick ASC LIMIT 1", (self.rev,))
            row = cur.fetchone()
        if not row:
            return "· todas las células verificadas en la revisión actual"
        cell = dict(row)

        user = f'Formula una pregunta con la que se buscaría este dato: "{cell["text"]}". Devuelve {{"pregunta": "..."}}'
        try:
            obj = self.llm.ask_json(SYSTEM, user)
        except Exception:
            return "· LLM no disponible para el autotest; se reintentará"
        q = obj.get("pregunta") if isinstance(obj, dict) else None
        if not q:
            self._mark_tested(cell["id"])
            return "· no se pudo formular una pregunta de prueba"

        res = self.recall(str(q), reinforce=False, navigate=False)
        self._mark_tested(cell["id"])
        if res["found"] and cell["id"] in res["cells"]:
            self.stats["selftest_ok"] = self.stats.get("selftest_ok", 0) + 1
            self._save_meta()
            return f"OK: «{q}»"

        self.stats["selftest_fail"] = self.stats.get("selftest_fail", 0) + 1
        repaired = self._reindex_cell(cell["id"])
        if repaired:
            self.stats["selftest_recovered"] = self.stats.get("selftest_recovered", 0) + 1
        self._save_meta()
        return f"FALLO: «{q}» no recuperó [{cell['id']}]" + (" - índice FTS5 reparado" if repaired else "")

    def task_reevaluate(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT id, text, importance, access_count 
                FROM cells 
                WHERE archived = 0 AND access_count >= 3 AND importance < 9.0 
                ORDER BY access_count DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                cid = row["id"]
                new_imp = min(10.0, row["importance"] + max(0.15, (10.0 - row["importance"]) / 5.0))
                cur.execute("UPDATE cells SET importance = ?, access_count = 0, updated_tick = ? WHERE id = ?",
                            (new_imp, self.tick, cid))
                self.conn.commit()
                self._bump()
                return f"reforzada célula [{cid}] a imp {new_imp:.1f} por uso intensivo"

            cur.execute("""
                SELECT id, text, importance, updated_tick 
                FROM cells 
                WHERE archived = 0 AND (? - updated_tick) >= ? 
                  AND eff(importance, updated_tick, ?, ?) BETWEEN ? AND (? + 0.35)
                ORDER BY updated_tick ASC LIMIT 1
            """, (self.tick, _cfg.MEM_MIN_AGE, self.tick, _cfg.MEM_TAU, _cfg.MEM_ARCHIVE_BELOW, _cfg.MEM_ARCHIVE_BELOW))
            cand = cur.fetchone()

        if cand:
            prompt = (
                f"Evalúa si este conocimiento merece ser preservado de forma duradera:\n"
                f"«{cand['text']}»\n"
                f'Devuelve JSON: {{"preserve": true/false, "importance": <1 a 10>}}'
            )
            try:
                res = self.llm.ask_json(SYSTEM, prompt)
                if res.get("preserve"):
                    boosted = clamp(res.get("importance", 6.5), 5.0, 10.0, 6.5)
                    with self.lock:
                        cur = self.conn.cursor()
                        cur.execute("UPDATE cells SET importance = ?, updated_tick = ? WHERE id = ?",
                                    (boosted, self.tick, cand["id"]))
                        self.conn.commit()
                    self._bump()
                    return f"rescate homeostático: [{cand['id']}] reevaluada a imp {boosted:.1f}"
            except Exception:
                pass

        return "· importancias y prioridades en equilibrio"

    def task_split(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, name, parent_id, level, celulas FROM nodes WHERE celulas > ? "
                        "ORDER BY celulas DESC LIMIT 1", (_cfg.MEM_MAX_CHILDREN,))
            overloaded = cur.fetchone()

        if not overloaded:
            return f"· factor de ramificación óptimo (≤{_cfg.MEM_MAX_CHILDREN} por nodo)"

        nid, nname, pid, lvl = overloaded["id"], overloaded["name"], overloaded["parent_id"], int(overloaded["level"])
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, text FROM cells WHERE node_id = ? AND archived = 0", (nid,))
            cells = [dict(r) for r in cur.fetchall()]

        if len(cells) < 4:
            return "· células insuficientes para partición"

        listing = "\n".join(f"- [{c['id']}] {c['text'][:80]}" for c in cells[:16])
        prompt = (
            f"El grupo de memoria '{nname}' tiene demasiados recuerdos ({len(cells)}).\n"
            f"Divídelos en 2 subgrupos temáticos distintos y coherentes:\n"
            f"{listing}\n\n"
            'Devuelve EXCLUSIVAMENTE este JSON: {"grupo_A": {"nombre": "...", "ids": ["id1"]}, "grupo_B": {"nombre": "...", "ids": ["id2"]}}'
        )
        try:
            res = self.llm.ask_json(SYSTEM, prompt)
            g_a = res.get("grupo_A") or {}
            g_b = res.get("grupo_B") or {}
            valid_ids = {c["id"] for c in cells}
            ids_b = [str(x) for x in (g_b.get("ids") or []) if str(x) in valid_ids]
            if len(ids_b) >= len(cells):
                ids_b = ids_b[: len(cells) // 2]
            name_b = str(g_b.get("nombre") or "").strip()[:60]
            if not (name_b and ids_b):
                return "· análisis de ramificación completado"

            as_children = lvl < N - 1
            new_parent, new_level = (nid, lvl + 1) if as_children else (pid, lvl)
            b_id = self._make_node(new_parent, {"name": name_b, "description": "Subdivisión por densidad"}, new_level)
            if b_id == nid:
                return "· partición descartada (el nuevo grupo coincide con el propio nodo)"
            groups = [(b_id, ids_b)]
            if as_children:
                name_a = str(g_a.get("nombre") or "").strip()[:60]
                taken = set(ids_b)
                ids_a = [str(x) for x in (g_a.get("ids") or []) if str(x) in valid_ids and str(x) not in taken]
                if name_a and ids_a and slugify(name_a) != slugify(name_b):
                    a_id = self._make_node(nid, {"name": name_a, "description": "Subdivisión por densidad"}, new_level)
                    if a_id != nid:
                        groups.append((a_id, ids_a))

            with self.lock:
                cur = self.conn.cursor()
                for target, ids in groups:
                    marks = ",".join("?" for _ in ids)
                    cur.execute(f"UPDATE cells SET node_id = ? WHERE node_id = ? AND id IN ({marks})",
                                [target, nid] + ids)
                for target in [nid] + [g[0] for g in groups]:
                    cur.execute("UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells "
                                "WHERE node_id = ? AND archived = 0) WHERE id = ?", (target, target))
                self.conn.commit()
            for target in [nid] + [g[0] for g in groups]:
                self._mark_dirty(target)
            self._bump()
            kind = "hijo" if as_children else "hermano"
            return f"partición exitosa en '{nname}': creado {kind} '{name_b}' ({len(ids_b)} células migradas)"
        except Exception as e:
            return f"aviso en partición: {e}"

    def task_merge(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT n1.id as id1, n2.id as id2, n1.name as name1, n2.name as name2, n1.parent_id
                FROM nodes n1
                JOIN nodes n2 ON n1.parent_id = n2.parent_id AND n1.id < n2.id
                WHERE n1.celulas <= 1 AND n2.celulas <= 1 AND n1.pinned = 0 AND n2.pinned = 0
                  AND NOT EXISTS (SELECT 1 FROM nodes c WHERE c.parent_id = n1.id)
                  AND NOT EXISTS (SELECT 1 FROM nodes c WHERE c.parent_id = n2.id)
                LIMIT 20
            """)
            candidates = cur.fetchall()

        if not candidates:
            return "· ramas integradas y sin hojas dispersas"

        emb = getattr(self, "emb", None)
        min_sim = float(getattr(_cfg, "MEM_MERGE_MIN_SIM", 0.50))
        best_sim = -1.0
        best_pair = None

        if emb is not None and emb.available():
            scored_pairs = []
            for p in candidates:
                sim = self._cos_nodes(p["id1"], p["id2"])
                scored_pairs.append((sim, p))
            scored_pairs.sort(key=lambda x: x[0], reverse=True)
            best_sim, best_pair = scored_pairs[0]
            if best_sim < min_sim:
                return (f"· ramas evaluadas ({len(candidates)} pares); sin afinidad semántica "
                        f"suficiente para fusión (máx sim={best_sim:.2f} < {min_sim:.2f})")
        else:
            best_pair = candidates[0]

        pair = best_pair
        name1, name2 = pair["name1"], pair["name2"]
        id1, id2 = pair["id1"], pair["id2"]
        sim_note = f" (similitud semántica detectada: {best_sim:.2f})" if best_sim >= 0 else ""
        prompt = (
            f"¿Pertenecen estos dos subtemas a un mismo concepto integrable?{sim_note}\n"
            f"Tema 1: {name1}\nTema 2: {name2}\n\n"
            "Reglas estrictas:\n"
            "1. Solo responde mergeable=true si representan el MISMO ámbito específico o variantes muy cercanas "
            "(ej: 'postres' y 'repostería', 'perros' y 'caninos').\n"
            "2. Si son temas distintos (ej: 'mascotas' y 'finanzas', 'salud' y 'recetas'), devuelve mergeable=false.\n"
            "3. PROHIBIDO crear categorías paraguas artificiales (como 'estilo de vida', 'varios' o 'gestión del hogar').\n"
            'Devuelve JSON: {"mergeable": true/false, "merged_name": "..."}'
        )
        try:
            res = self.llm.ask_json(SYSTEM, prompt)
            if res.get("mergeable"):
                new_name = str(res.get("merged_name") or name1).strip()[:60]
                with self.lock:
                    cur = self.conn.cursor()
                    cur.execute("UPDATE cells SET node_id = ? WHERE node_id = ?", (id1, id2))
                    cur.execute("DELETE FROM nodes WHERE id = ?", (id2,))
                    cur.execute("UPDATE nodes SET name = ?, dirty = 1 WHERE id = ?", (new_name, id1))
                    cur.execute("UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells WHERE node_id = ? AND archived = 0) WHERE id = ?", (id1, id1))
                    self.conn.commit()
                self._mark_dirty(id1)
                self._bump()
                extra = f" (sim={best_sim:.2f})" if best_sim >= 0 else ""
                return f"fusión de ramas: '{name1}' + '{name2}'{extra} → '{new_name}'"
        except Exception:
            pass

        return "· ramas evaluadas; sin fusiones requeridas"

    def task_embed(self) -> str:
        emb = getattr(self, "emb", None)
        if emb is None:
            return "· embeddings desactivados"
        return emb.sync()

    def _cos_nodes(self, a: int | str, b: int | str) -> float:
        emb = getattr(self, "emb", None)
        if emb is None:
            return 0.0
        try:
            return emb.cos_nodes(a, b)
        except Exception:
            return 0.0

    def _cos_cells(self, a: str, b: str) -> float:
        emb = getattr(self, "emb", None)
        if emb is None:
            return 0.0
        try:
            return emb.cos_cells(a, b)
        except Exception:
            return 0.0
