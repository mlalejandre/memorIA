# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .common import *
from .db import DatabaseMixin
from .storage import StorageMixin
from .retrieval import RetrievalMixin
from .tasks import HomeostasisMixin
from .ingestion import IngestionMixin
from .explorer import ExplorerMixin
from .llm import MemLLM

try:
    import episodica as _epi
except ImportError:
    _epi = None
try:
    import vectores as _vectores
except ImportError:
    _vectores = None
try:
    import rerank as _rerank
except ImportError:
    _rerank = None


class MemoryTree(DatabaseMixin, StorageMixin, RetrievalMixin, HomeostasisMixin, IngestionMixin, ExplorerMixin):
    """Árbol de memoria jerárquica unificada de profundidad variable."""

    TASKS = ["repair", "embed", "decay", "describe", "dedupe", "selftest", "reevaluate", "split", "merge"]

    def __init__(self, root: Path, llm: MemLLM) -> None:
        self.root = Path(root).resolve()
        self.db_path = self.root / "memoria.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.lock = threading.RLock()
        self._thread_local = threading.local()

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row

        with self.lock:
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA synchronous = NORMAL;")
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.execute("PRAGMA busy_timeout = 30000;")
            self.conn.create_function("eff", 4, self._sql_eff)

        self._init_db()
        self._load_meta()

        self.episodes = None
        if _epi is not None and _epi.enabled():
            try:
                self.episodes = _epi.EpisodicMemory(self)
            except Exception:
                self.episodes = None

        self.emb = None
        if _vectores is not None and _vectores.enabled():
            try:
                self.emb = _vectores.EmbeddingService(self)
            except Exception:
                self.emb = None

        self.reranker = None  # activo solo si RERANK_MODE = llm | cross
        if _rerank is not None:
            try:
                self.reranker = _rerank.Reranker(self)
            except Exception:
                self.reranker = None

        self._init_ingest_queue()

    def advance(self, n: int = 1) -> int:
        self.tick += max(1, int(n))
        self.step += 1
        self._save_meta()
        return self.tick

    def _bump(self) -> None:
        self.rev += 1
        self._save_meta()

    def shutdown(self) -> None:
        stop = getattr(self, "_ingest_stop", None)
        if stop is not None:
            stop.set()
        t = getattr(self, "_ingest_thread", None)
        if t and t.is_alive():
            t.join(timeout=2.0)
        try:
            if hasattr(self, "conn") and hasattr(self, "lock"):
                with self.lock:
                    self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except (sqlite3.Error, Exception):
            pass

    def close(self) -> None:
        self.shutdown()
        if hasattr(self._thread_local, "conn") and self._thread_local.conn:
            try:
                self._thread_local.conn.close()
            except Exception:
                pass
            self._thread_local.conn = None
        if hasattr(self, "conn"):
            try:
                if hasattr(self, "lock"):
                    with self.lock:
                        self.conn.close()
                else:
                    self.conn.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()
