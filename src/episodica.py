# Memoria episódica: registro literal y buscable (SQLite + FTS5).
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any

import config as C


def enabled() -> bool:
    return bool(getattr(C, "EPI_ENABLED", True))


def fold(text: Any) -> str:
    t = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def stem(word: str) -> str:
    w = fold(word)
    return w[: max(4, len(w) - 3)] if len(w) >= 6 else w


class EpisodicMemory:
    """Guarda cada mensaje tal cual (nada se resume ni se olvida) en la misma memoria.db."""

    BACKFILL_KEY = "episodes_backfill_v1"

    def __init__(self, tree: Any) -> None:
        self.tree = tree
        self._init_db()

    def _init_db(self) -> None:
        with self.tree.lock:
            conn = self.tree.conn
            conn.execute(
                "CREATE TABLE IF NOT EXISTS episodes ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT UNIQUE, ts REAL NOT NULL, "
                "role TEXT NOT NULL, text TEXT NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts)")
            # El texto se indexa sin acentos y en minúsculas: así 'riñón' encuentra 'rinones'.
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5("
                "ep_id UNINDEXED, text, tokenize = 'unicode61')"
            )
            conn.commit()
            n_ep = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            n_fts = conn.execute("SELECT COUNT(*) FROM episodes_fts").fetchone()[0]
            if n_ep != n_fts:
                self._rebuild()

    def _rebuild(self) -> None:
        with self.tree.lock:
            conn = self.tree.conn
            conn.execute("DELETE FROM episodes_fts")
            rows = conn.execute("SELECT id, text FROM episodes").fetchall()
            conn.executemany("INSERT INTO episodes_fts (ep_id, text) VALUES (?, ?)",
                             [(r[0], fold(r[1])) for r in rows])
            conn.commit()

    def add(self, role: str, text: str, ts: float | None = None) -> int | None:
        text = str(text or "").strip()
        if not text:
            return None
        role = role if role in ("user", "assistant") else (str(role)[:20] or "user")
        ts = float(ts if ts is not None else time.time())
        uid = hashlib.sha1(f"{ts:.3f}|{role}|{text}".encode("utf-8")).hexdigest()
        with self.tree.lock:
            conn = self.tree.conn
            cur = conn.execute("INSERT OR IGNORE INTO episodes (uid, ts, role, text) VALUES (?, ?, ?, ?)",
                               (uid, ts, role, text))
            if cur.rowcount <= 0:
                conn.commit()
                return None
            eid = cur.lastrowid
            conn.execute("INSERT INTO episodes_fts (ep_id, text) VALUES (?, ?)", (eid, fold(text)))
            conn.commit()
        return eid

    def search(self, question: str, limit: int = 8) -> list[dict[str, Any]]:
        terms: list[str] = []
        for w in self.tree._fts_terms(question, limit=12):
            s = stem(w)
            if s and s not in terms:
                terms.append(s)
        if not terms:
            return []
        query = " OR ".join("%s*" % t.replace('"', "") for t in terms)
        with self.tree.lock:
            try:
                rows = self.tree.conn.execute(
                    "SELECT e.id, e.ts, e.role, e.text, bm25(episodes_fts) AS rank FROM episodes_fts "
                    "JOIN episodes e ON e.id = CAST(episodes_fts.ep_id AS INTEGER) "
                    "WHERE episodes_fts MATCH ? ORDER BY rank ASC LIMIT ?",
                    (query, int(limit)),
                ).fetchall()
            except sqlite3.Error:
                return []
        return [{"id": r["id"], "ts": r["ts"], "role": r["role"], "text": r["text"],
                 "score": round(-float(r["rank"]), 3)} for r in rows]

    def count(self) -> int:
        with self.tree.lock:
            return int(self.tree.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])

    def backfill(self, directory: Any) -> int:
        """Importa (una sola vez) lo que ya había en sesion.json y _archivo.jsonl."""
        with self.tree.lock:
            done = self.tree.conn.execute("SELECT value FROM meta WHERE key = ?",
                                          (self.BACKFILL_KEY,)).fetchone()
        if done:
            return 0
        d = Path(directory)
        items: list[tuple[float, str, str]] = []

        def take(e: Any) -> None:
            if isinstance(e, dict) and e.get("role") in ("user", "assistant") and e.get("text"):
                try:
                    items.append((float(e.get("ts") or 0.0), e["role"], str(e["text"])))
                except (TypeError, ValueError):
                    pass

        try:
            for line in (d / "_archivo.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    take(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass
        try:
            data = json.loads((d / "sesion.json").read_text(encoding="utf-8"))
            for e in (data.get("entries", []) if isinstance(data, dict) else []):
                take(e)
        except (OSError, ValueError):
            pass

        n = 0
        for ts, role, text in sorted(items):
            if self.add(role, text, ts or None):
                n += 1
        with self.tree.lock:
            self.tree.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                                   (self.BACKFILL_KEY, str(int(time.time()))))
            self.tree.conn.commit()
        return n
