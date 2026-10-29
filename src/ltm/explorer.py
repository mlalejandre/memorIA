# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any

from .common import N, slugify

try:
    import vectores as _vectores
except ImportError:
    _vectores = None
try:
    import descomponer as _decomp  # [PARCHE19]
except ImportError:
    _decomp = None
try:
    import refinar as _refinar  # [PARCHE20]
    import config as _cfg
except ImportError:
    _refinar = None


class ExplorerMixin:
    """Interacción con la interfaz web: inspección, estadísticas, CRUD y exportación."""

    def counts(self) -> dict[str, int]:
        try:
            rconn = self._get_reader()
            cur = rconn.cursor()
            cur.execute(
                "SELECT (SELECT COUNT(*) FROM cells WHERE archived = 0), "
                "(SELECT COUNT(*) FROM cells WHERE archived = 1), "
                "(SELECT COUNT(*) FROM nodes), (SELECT COUNT(*) FROM nodes WHERE dirty = 1)"
            )
            alive, archived, nodes, dirty = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM ingest_queue WHERE status = 'pending'")
            q_pending = cur.fetchone()[0]
        except Exception:
            with self.lock:
                cur = self.conn.cursor()
                cur.execute(
                    "SELECT (SELECT COUNT(*) FROM cells WHERE archived = 0), "
                    "(SELECT COUNT(*) FROM cells WHERE archived = 1), "
                    "(SELECT COUNT(*) FROM nodes), (SELECT COUNT(*) FROM nodes WHERE dirty = 1)"
                )
                alive, archived, nodes, dirty = cur.fetchone()
                try:
                    cur.execute("SELECT COUNT(*) FROM ingest_queue WHERE status = 'pending'")
                    q_pending = cur.fetchone()[0]
                except Exception:
                    q_pending = self._ingest_q.qsize()

        return {
            "alive": alive, "archived": archived, "nodes": nodes, "dirty": dirty,
            "ingest_queue": max(self._ingest_q.qsize(), q_pending),
            "episodes": self.episodes.count() if getattr(self, "episodes", None) else 0,
            "vectors": self.emb.count() if getattr(self, "emb", None) else 0,
            "synapses": self._get_synapses_count(),
        }

    def render(self) -> str:
        lines: list[str] = []
        with self.lock:
            cur = self.conn.cursor()

            def rec(parent_id: int | None, depth: int) -> None:
                cur.execute(
                    "SELECT * FROM nodes WHERE (parent_id IS ? OR parent_id = ?) ORDER BY importance DESC, name ASC",
                    (parent_id, parent_id),
                )
                nodes = cur.fetchall()
                for node in nodes:
                    label = f"nivel {depth + 1}"
                    lines.append("  " * depth + f"▸ {label}: {node['name']} [imp {node['importance']:.1f}] — {node['description']}")
                    cur.execute("SELECT * FROM cells WHERE node_id = ? AND archived = 0", (node["id"],))
                    for c in cur.fetchall():
                        lines.append("  " * (depth + 1) + f"• ({self.eff(c):.1f}) {c['text']}")
                    if depth < N - 1:
                        rec(node["id"], depth + 1)

            rec(None, 0)
        return "\n".join(lines) if lines else "(memoria vacía)"

    def summary(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM cells WHERE archived = 0")
            alive = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM cells WHERE archived = 1")
            archived = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM nodes")
            nodes = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM nodes WHERE dirty = 1")
            dirty = cur.fetchone()[0]
        return (
            f"tick={self.tick} rev={self.rev} | nodos={nodes} (sin describir: {dirty}) | células={alive} | "
            f"archivadas={archived} | motor=SQLite-FTS5 (BM25 nativo) | "
            f"autotests OK={self.stats.get('selftest_ok', 0)} | recall_híbrido={self.stats.get('recall_hibrido', 0)}"
        )

    def get_graph_data(self) -> dict[str, Any]:
        try:
            rconn = self._get_reader()
            cur = rconn.cursor()
            cur.execute("""
                SELECT id, parent_id, level, slug, name, description, importance, celulas
                FROM nodes ORDER BY level ASC, name ASC
            """)
            nodes = [dict(r) for r in cur.fetchall()]
            cur.execute("""
                SELECT source_id, target_id, weight, coactivations, last_tick
                FROM synapses
                WHERE source_id < target_id
                ORDER BY weight DESC
            """)
            synapses = [dict(r) for r in cur.fetchall()]
            return {"nodes": nodes, "synapses": synapses}
        except Exception:
            with self.lock:
                cur = self.conn.cursor()
                cur.execute("""
                    SELECT id, parent_id, level, slug, name, description, importance, celulas
                    FROM nodes ORDER BY level ASC, name ASC
                """)
                nodes = [dict(r) for r in cur.fetchall()]
                try:
                    cur.execute("""
                        SELECT source_id, target_id, weight, coactivations, last_tick
                        FROM synapses
                        WHERE source_id < target_id
                        ORDER BY weight DESC
                    """)
                    synapses = [dict(r) for r in cur.fetchall()]
                except Exception:
                    synapses = []
                return {"nodes": nodes, "synapses": synapses}

    def _get_synapses_count(self) -> int:
        try:
            rconn = self._get_reader()
            return int(rconn.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]) // 2
        except Exception:
            return 0

    def get_explorer_tree(self) -> list[dict[str, Any]]:
        try:
            rconn = self._get_reader()
            cur = rconn.cursor()
            cur.execute("""
                SELECT id, parent_id, level, slug, name, description, importance, celulas, dirty
                FROM nodes
                ORDER BY level ASC, importance DESC, name ASC
            """)
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            with self.lock:
                cur = self.conn.cursor()
                cur.execute("""
                    SELECT id, parent_id, level, slug, name, description, importance, celulas, dirty
                    FROM nodes
                    ORDER BY level ASC, importance DESC, name ASC
                """)
                return [dict(r) for r in cur.fetchall()]

    def get_node_cells(self, node_id: int) -> list[dict[str, Any]]:
        try:
            rconn = self._get_reader()
            cur = rconn.cursor()
            cur.execute("""
                SELECT id, node_id, text, importance, kind, fuente, ingest_timestamp,
                       created_tick, updated_tick, access_count, archived
                FROM cells
                WHERE node_id = ?
                ORDER BY archived ASC, importance DESC, updated_tick DESC
            """, (node_id,))
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            with self.lock:
                cur = self.conn.cursor()
                cur.execute("""
                    SELECT id, node_id, text, importance, kind, fuente, ingest_timestamp,
                           created_tick, updated_tick, access_count, archived
                    FROM cells
                    WHERE node_id = ?
                    ORDER BY archived ASC, importance DESC, updated_tick DESC
                """, (node_id,))
                return [dict(r) for r in cur.fetchall()]

    def save_cell(self, cell_id: str | None, node_id: int, text: str, importance: float, kind: str, fuente: str) -> dict[str, Any]:
        with self.lock:
            cur = self.conn.cursor()
            if cell_id:
                cur.execute("""
                    UPDATE cells 
                    SET text = ?, importance = ?, kind = ?, fuente = ?, updated_tick = ?
                    WHERE id = ?
                """, (text.strip(), importance, kind, fuente, self.tick, cell_id))
            else:
                cell_id = f"{self.tick:07d}-{secrets.token_hex(6)}"
                cur.execute("""
                    INSERT INTO cells (id, node_id, text, importance, kind, fuente, ingest_timestamp, created_tick, updated_tick)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (cell_id, node_id, text.strip(), importance, kind, fuente, time.strftime("%Y-%m-%d %H:%M:%S"), self.tick, self.tick))
                cur.execute("UPDATE nodes SET celulas = celulas + 1 WHERE id = ?", (node_id,))
            self.conn.commit()
            self._mark_dirty(node_id)
            self._bump()
            return {"ok": True, "id": cell_id}

    def delete_cell(self, cell_id: str) -> bool:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT node_id FROM cells WHERE id = ?", (cell_id,))
            row = cur.fetchone()
            if not row:
                return False
            node_id = row[0]
            cur.execute("DELETE FROM cells WHERE id = ?", (cell_id,))
            cur.execute("UPDATE nodes SET celulas = MAX(0, celulas - 1) WHERE id = ?", (node_id,))
            self.conn.commit()
            self._mark_dirty(node_id)
            self._bump()
            return True

    def save_node(self, node_id: int | None, parent_id: int | None, name: str, description: str) -> dict[str, Any]:
        with self.lock:
            cur = self.conn.cursor()
            slug = slugify(name)
            if node_id:
                cur.execute("""
                    UPDATE nodes 
                    SET name = ?, slug = ?, description = ?, dirty = 1
                    WHERE id = ?
                """, (name.strip(), slug, description.strip(), node_id))
            else:
                level = 0
                if parent_id is not None:
                    cur.execute("SELECT level FROM nodes WHERE id = ?", (parent_id,))
                    r = cur.fetchone()
                    level = min(N - 1, r[0] + 1) if r else 0
                cur.execute("""
                    INSERT INTO nodes (parent_id, level, slug, name, description, dirty, created_tick)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                """, (parent_id, level, slug, name.strip(), description.strip(), self.tick))
                node_id = cur.lastrowid
                cur.execute("UPDATE nodes SET pinned = 1 WHERE id = ?", (node_id,))
            self.conn.commit()
            self._bump()
            return {"ok": True, "id": node_id}

    def delete_node(self, node_id: int) -> bool:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            self.conn.commit()
            self._bump()
            return True

    def export_to_filesystem(self, target_base_dir: Path) -> int:
        target_base_dir = Path(target_base_dir)
        target_base_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM nodes")
            nodes = {r["id"]: dict(r) for r in cur.fetchall()}
            
            def get_full_folder(nid):
                parts = []
                curr = nid
                while curr is not None and curr in nodes:
                    parts.append(nodes[curr]["slug"])
                    curr = nodes[curr]["parent_id"]
                return target_base_dir.joinpath(*reversed(parts))

            for nid, n in nodes.items():
                f_path = get_full_folder(nid)
                f_path.mkdir(parents=True, exist_ok=True)
                n_info = {
                    "name": n["name"], "description": n["description"],
                    "dirty": bool(n["dirty"]), "importance": n["importance"],
                    "created_tick": n["created_tick"], "celulas": n["celulas"]
                }
                (f_path / "_nodo.json").write_text(json.dumps(n_info, ensure_ascii=False, indent=2), encoding="utf-8")

            cur.execute("SELECT * FROM cells")
            for c in cur.fetchall():
                f_path = get_full_folder(c["node_id"]) if not c["archived"] else target_base_dir / "_archivo"
                f_path.mkdir(parents=True, exist_ok=True)
                c_info = dict(c)
                (f_path / f"c-{c['id']}.json").write_text(json.dumps(c_info, ensure_ascii=False, indent=2), encoding="utf-8")
                count += 1
        return count

    def log_episode(self, role: str, text: str, ts: float | None = None) -> int | None:
        if getattr(self, "episodes", None) is None:
            return None
        return self.episodes.add(role, text, ts)

    def search_episodes(self, question: str, limit: int = 8,
                        extra_queries: list[str] | None = None,
                        diversify: bool | None = None) -> list[dict[str, Any]]:
        if getattr(self, "episodes", None) is None:
            return []
        if diversify is None:  # [PARCHE20] diversidad por día: candidatos más hondos y tope por día
            diversify = _refinar is not None and _refinar.diversity_mode() == "dia"
        if diversify and _refinar is not None:
            pool = max(1, int(getattr(_cfg, "EPI_DIVERSITY_POOL", 3))) * limit
            return _refinar.por_dia(self.search_episodes(question, pool, extra_queries, diversify=False), limit)
        if extra_queries and _decomp is not None:  # [PARCHE19] una búsqueda por subconsulta y fusión RRF
            lists = [self.search_episodes(q, limit, diversify=False) for q in [question, *extra_queries]]
            return _decomp.fuse_many(lists, limit)
        lex = self.episodes.search(question, limit)
        emb = getattr(self, "emb", None)
        sem: list[dict[str, Any]] = []
        if emb is not None:
            try:
                sem = emb.search_episodes(question, limit)
            except Exception:
                sem = []
        return _vectores.fuse(lex, sem, limit) if sem and _vectores is not None else lex

    def backfill_episodes(self, directory: Any) -> int:
        if getattr(self, "episodes", None) is None:
            return 0
        return self.episodes.backfill(directory)

    def process_user_message(self, message: str) -> str:
        msg = message.strip()
        if not msg:
            return ""
        if msg.endswith("?") or any(w in msg.lower() for w in ["qué", "que", "cómo", "como", "quién", "quien", "cuándo", "cuando", "dónde", "donde", "por qué", "cuál", "cual"]):
            res = self.recall(msg, reinforce=True)
            if res.get("found"):
                srcs = f"\n📚 Fuentes: {', '.join(res['sources'])}" if res.get("sources") else ""
                return f"💬 {res['answer']}{srcs} (Ruta: {res.get('path', '-')})"
            return "💬 No encontré información sobre eso en la memoria a largo plazo."
        else:
            r = self.store(msg, importance=7, kind="dato")
            tag = "reforzado" if r.get("duplicate") else "memorizado"
            return f"💾 [{tag}] Guardado en: {r.get('path', 'raíz')} (ID: {r.get('id', '-')})"
