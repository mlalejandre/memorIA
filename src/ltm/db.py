# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import sqlite3
import threading
from typing import Any


class DatabaseMixin:
    """Gestión de esquemas SQLite, triggers, metadatos y conexiones por hilo."""

    def _init_db(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingest_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_tick INTEGER DEFAULT 0
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ingest_queue_status ON ingest_queue(status);")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS synapses (
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    coactivations INTEGER DEFAULT 1,
                    last_tick INTEGER NOT NULL,
                    created_tick INTEGER NOT NULL,
                    PRIMARY KEY (source_id, target_id),
                    FOREIGN KEY(source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_synapses_lookup ON synapses(source_id, weight DESC);")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER,
                    level INTEGER NOT NULL,
                    slug TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    importance REAL DEFAULT 5.0,
                    celulas INTEGER DEFAULT 0,
                    dirty INTEGER DEFAULT 1,
                    created_tick INTEGER DEFAULT 0,
                    FOREIGN KEY(parent_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_level ON nodes(level);")
            if "pinned" not in {r[1] for r in cur.execute("PRAGMA table_info(nodes)")}:
                cur.execute("ALTER TABLE nodes ADD COLUMN pinned INTEGER DEFAULT 0")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS cells (
                    id TEXT PRIMARY KEY,
                    node_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    importance REAL NOT NULL,
                    kind TEXT DEFAULT 'dato',
                    fuente TEXT,
                    ingest_timestamp TEXT,
                    created_tick INTEGER NOT NULL,
                    updated_tick INTEGER NOT NULL,
                    last_access_tick INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0,
                    ruta_original TEXT,
                    tested_rev INTEGER DEFAULT -1,
                    tested_tick INTEGER DEFAULT 0,
                    FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cells_node ON cells(node_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cells_archived ON cells(archived);")

            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS cells_fts USING fts5(
                    cell_id UNINDEXED,
                    text,
                    tokenize = 'unicode61'
                );
            """)

            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_cells_ai AFTER INSERT ON cells BEGIN
                    INSERT INTO cells_fts(cell_id, text) VALUES (new.id, new.text);
                END;
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_cells_ad AFTER DELETE ON cells BEGIN
                    DELETE FROM cells_fts WHERE cell_id = old.id;
                END;
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_cells_au AFTER UPDATE OF text ON cells BEGIN
                    DELETE FROM cells_fts WHERE cell_id = old.id;
                    INSERT INTO cells_fts(cell_id, text) VALUES (new.id, new.text);
                END;
            """)

            cell_columns = {r[1] for r in cur.execute("PRAGMA table_info(cells)")}
            for col in ("valid_from", "valid_until", "superseded_by"):
                if col not in cell_columns:
                    cur.execute(f"ALTER TABLE cells ADD COLUMN {col} TEXT")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cells_validity ON cells(valid_until, superseded_by)")
            cur.execute("UPDATE cells SET valid_from = COALESCE(NULLIF(ingest_timestamp, ''), strftime('%Y-%m-%d %H:%M:%S','now')) "
                        "WHERE valid_from IS NULL OR valid_from = ''")
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_cells_valid_from_ai
                AFTER INSERT ON cells
                WHEN NEW.valid_from IS NULL OR NEW.valid_from = ''
                BEGIN
                    UPDATE cells SET valid_from = COALESCE(NULLIF(NEW.ingest_timestamp, ''),
                        strftime('%Y-%m-%d %H:%M:%S','now')) WHERE id = NEW.id;
                END;
            """)
            self.conn.commit()

    def _load_meta(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT key, value FROM meta")
            rows = dict(cur.fetchall())
            self.tick = int(rows.get("tick", 0))
            self.step = int(rows.get("step", 0))
            self.rev = int(rows.get("rev", 0))
            try:
                self.stats = json.loads(rows.get("stats", "{}"))
            except Exception:
                self.stats = {}
            for k in ["selftest_ok", "selftest_fail", "selftest_recovered", "recall_exact",
                      "recall_lexico", "recall_semantica", "recall_embeddings", "recall_arbol", "recall_hibrido"]:
                self.stats.setdefault(k, 0)

    def _save_meta(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('tick', ?)", (str(self.tick),))
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('step', ?)", (str(self.step),))
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('rev', ?)", (str(self.rev),))
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('stats', ?)", (json.dumps(self.stats),))
            self.conn.commit()

    def _get_reader(self) -> sqlite3.Connection:
        """Conexión SQLite aislada por hilo mediante threading.local()."""
        if not hasattr(self._thread_local, "conn") or self._thread_local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            self._thread_local.conn = conn
        return self._thread_local.conn

    @staticmethod
    def _sql_eff(imp: Any, updated_tick: Any, cur_tick: Any, tau: Any) -> float:
        try:
            i = float(imp)
            u = int(updated_tick)
            c = int(cur_tick)
            t = float(tau)
            age = max(0, c - u)
            return i * math.exp(-age / (t * max(i, 1.0) ** 3))
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
