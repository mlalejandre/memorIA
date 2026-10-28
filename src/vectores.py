# Embeddings (servidor OpenAI-compatible en la GPU2) + índice vectorial en SQLite.
from __future__ import annotations

import json
import math
import os
import sqlite3
import heapq
import sys
import time
import unicodedata
import urllib.request
from array import array
from collections import OrderedDict
from operator import mul
from typing import Any

import config as C

try:  # opcional: acelera mucho el coseno con muchas decenas de miles de vectores
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _under_tests() -> bool:
    argv0 = (sys.argv[0] if sys.argv else "").lower()
    return "unittest" in argv0 or "pytest" in argv0


def enabled() -> bool:
    """Desactivable con EMB_ENABLED=0 o HMEM_NO_EMB=1. Bajo unittest/pytest está apagado
    (los tests inyectan su propio embedder) salvo que se defina HMEM_TEST_EMB=1."""
    if os.getenv("HMEM_NO_EMB", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if not getattr(C, "EMB_ENABLED", True):
        return False
    if _under_tests() and not os.getenv("HMEM_TEST_EMB"):
        return False
    return True


def fold(text: Any) -> str:
    t = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _norm(v: Any) -> array:
    a = array("f", (float(x) for x in v))
    n = math.sqrt(sum(x * x for x in a)) or 1.0
    return array("f", (x / n for x in a))


def fuse(a: list[dict], b: list[dict], limit: int = 8, k: int = 60, key: str = "id") -> list[dict]:
    """Reciprocal Rank Fusion de dos listas de dicts."""
    scores: dict[Any, float] = {}
    by_id: dict[Any, dict] = {}
    for lst in (a, b):
        for rank, it in enumerate(lst):
            scores[it[key]] = scores.get(it[key], 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(it[key], it)
    order = sorted(scores, key=lambda i: scores[i], reverse=True)[:limit]
    return [by_id[i] for i in order]


class Embedder:
    """Cliente de /v1/embeddings (llama-server --embeddings, vLLM, Ollama...)."""

    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self.url = (url or C.EMB_URL).rstrip("/") + "/embeddings"
        self.model = model or C.EMB_MODEL_ALIAS
        self._fail_until = 0.0
        self.calls = 0

    def available(self) -> bool:
        return time.time() >= self._fail_until

    def embed(self, texts: list[str]) -> list[array] | None:
        """Devuelve un vector normalizado por texto, o None si el servidor no responde."""
        if not texts:
            return []
        if not self.available():
            return None
        payload = {"model": self.model, "input": [(t or " ")[:2000] for t in texts]}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=float(getattr(C, "EMB_TIMEOUT_S", 20))) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vecs = []
            for it in items:
                emb = it["embedding"]
                if not emb or isinstance(emb[0], list):
                    raise ValueError("embedding no plano (¿falta --pooling?)")
                vecs.append(_norm(emb))
            if len(vecs) != len(texts):
                raise ValueError("número de vectores distinto al de textos")
        except Exception:
            self._fail_until = time.time() + 20.0
            return None
        self.calls += 1
        return vecs


_TRIGGERS = (
    "CREATE TRIGGER IF NOT EXISTS trg_emb_cells_au AFTER UPDATE OF text ON cells BEGIN "
    "DELETE FROM embeddings WHERE kind = 'cell' AND ref = old.id; END;",
    "CREATE TRIGGER IF NOT EXISTS trg_emb_cells_ad AFTER DELETE ON cells BEGIN "
    "DELETE FROM embeddings WHERE kind = 'cell' AND ref = old.id; END;",
    "CREATE TRIGGER IF NOT EXISTS trg_emb_nodes_au AFTER UPDATE OF name, description ON nodes BEGIN "
    "DELETE FROM embeddings WHERE kind = 'node' AND ref = CAST(old.id AS TEXT); END;",
    "CREATE TRIGGER IF NOT EXISTS trg_emb_nodes_ad AFTER DELETE ON nodes BEGIN "
    "DELETE FROM embeddings WHERE kind = 'node' AND ref = CAST(old.id AS TEXT); END;",
)


class VectorStore:
    """Vectores float32 en SQLite + caché en memoria (matriz numpy o lista de arrays)."""

    def __init__(self, tree: Any) -> None:
        self.tree = tree
        self._cache: dict[str, tuple[Any, list[str], Any, int]] = {}
        self.incremental_appends = 0   # [PARCHE8] veces que la caché creció sin releer todos los vectores
        self.full_rebuilds = 0
        self._init_db()

    def _init_db(self) -> None:
        with self.tree.lock:
            c = self.tree.conn
            c.execute("CREATE TABLE IF NOT EXISTS embeddings ("
                      "kind TEXT NOT NULL, ref TEXT NOT NULL, model TEXT NOT NULL, "
                      "dim INTEGER NOT NULL, vec BLOB NOT NULL, PRIMARY KEY (kind, ref))")
            c.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_kind ON embeddings(kind)")
            # [PARCHE8] índice cubriente: el COUNT por (kind, model) ya no lee todos los vectores del disco
            c.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_kind_model ON embeddings(kind, model)")
            for sql in _TRIGGERS:  # un texto editado invalida su vector automáticamente
                c.execute(sql)
            c.commit()

    def put_many(self, rows: list[tuple[str, str, str, array]]) -> None:
        with self.tree.lock:
            self.tree.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (kind, ref, model, dim, vec) VALUES (?, ?, ?, ?, ?)",
                [(k, str(r), m, len(v), v.tobytes()) for k, r, m, v in rows])
            self.tree.conn.commit()

    def put(self, kind: str, ref: Any, model: str, vec: array) -> None:
        self.put_many([(kind, str(ref), model, vec)])

    def count(self, kind: str | None = None) -> int:
        with self.tree.lock:
            if kind:
                return int(self.tree.conn.execute("SELECT COUNT(*) FROM embeddings WHERE kind = ?",
                                                  (kind,)).fetchone()[0])
            return int(self.tree.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def get_vectors(self, kind: str, refs: list[str], model: str) -> dict[str, array]:
        if not refs:
            return {}
        marks = ",".join("?" * len(refs))
        with self.tree.lock:
            rows = self.tree.conn.execute(
                f"SELECT ref, vec FROM embeddings WHERE kind = ? AND model = ? AND ref IN ({marks})",
                [kind, model, *[str(r) for r in refs]]).fetchall()
        out = {}
        for r in rows:
            a = array("f")
            a.frombytes(r[1])
            out[r[0]] = a
        return out

    def _matrix(self, kind: str, model: str):
        with self.tree.lock:
            sig = tuple(self.tree.conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM embeddings WHERE kind = ?", (kind,)).fetchone())
        key = (sig, model, np is not None)
        cached = self._cache.get(kind)
        if cached and cached[0] == key:
            return cached[1], cached[2], cached[3]
        # [PARCHE8] caché INCREMENTAL: si desde la última vez solo se han AÑADIDO vectores (nada borrado ni
        # reemplazado), se leen solo los nuevos y se agregan al final, en vez de releer toda la tabla.
        if cached and cached[0][1:] == key[1:] and cached[2] is not None:
            (old_cnt, old_max) = cached[0][0]
            if sig[0] > old_cnt and sig[1] > old_max:
                with self.tree.lock:
                    new_rows = self.tree.conn.execute(
                        "SELECT ref, dim, vec FROM embeddings WHERE kind = ? AND model = ? AND rowid > ? ORDER BY rowid",
                        (kind, model, old_max)).fetchall()
                dim = cached[3]
                if old_cnt + len(new_rows) == sig[0] and all(int(r[1]) == dim for r in new_rows):
                    refs = cached[1] + [r[0] for r in new_rows]
                    if np is not None:
                        add = np.frombuffer(b"".join(bytes(r[2]) for r in new_rows),
                                            dtype=np.float32).reshape(len(new_rows), dim)
                        mat = np.vstack([cached[2], add])
                    else:
                        mat = list(cached[2])
                        for r in new_rows:
                            a = array("f")
                            a.frombytes(bytes(r[2]))
                            mat.append(a)
                    self._cache[kind] = (key, refs, mat, dim)
                    self.incremental_appends += 1
                    return refs, mat, dim
        self.full_rebuilds += 1
        with self.tree.lock:
            rows = self.tree.conn.execute(
                "SELECT ref, dim, vec FROM embeddings WHERE kind = ? AND model = ?", (kind, model)).fetchall()
        refs: list[str] = []
        blobs: list[bytes] = []
        dim = int(rows[0][1]) if rows else 0
        for r in rows:
            if int(r[1]) == dim:
                refs.append(r[0])
                blobs.append(bytes(r[2]))
        if not refs:
            mat = None
        elif np is not None:
            mat = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(refs), dim)
        else:
            mat = []
            for b in blobs:
                a = array("f")
                a.frombytes(b)
                mat.append(a)
        self._cache[kind] = (key, refs, mat, dim)
        return refs, mat, dim

    # búsqueda exacta por bloques para conjuntos grandes.
    def _search_streaming(self, kind: str, qvec: array, k: int | None,
                          min_sim: float, model: str) -> list[tuple[str, float]]:
        """Escaneo exacto con RAM acotada; el coste temporal sigue siendo O(N·dim)."""
        chunk_size = max(32, int(os.getenv("HMEM_VECTOR_CHUNK_SIZE", "512")))
        cur = self.tree.conn.cursor()
        with self.tree.lock:
            cur.execute(
                "SELECT ref, dim, vec FROM embeddings WHERE kind = ? AND model = ?",
                (kind, model),
            )
        heap: list[tuple[float, int, str]] = []
        all_hits: list[tuple[str, float]] = [] if k is None else []
        sequence = 0
        keep = max(1, int(k)) if k is not None else None
        q_np = np.frombuffer(qvec, dtype=np.float32) if np is not None else None

        while True:
            with self.tree.lock:
                rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            valid = [r for r in rows if int(r[1]) == len(qvec)]
            if not valid:
                continue

            if np is not None:
                blobs = [bytes(r[2]) for r in valid]
                try:
                    block = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(valid), len(qvec))
                    sims = block @ q_np
                except (ValueError, TypeError):
                    # Un registro dañado no debe tumbar la recuperación completa.
                    continue
                local_keep = len(valid) if keep is None else min(keep, len(valid))
                if local_keep < len(valid):
                    indexes = np.argpartition(-sims, local_keep - 1)[:local_keep]
                else:
                    indexes = range(len(valid))
                scored = ((str(valid[int(i)][0]), float(sims[int(i)])) for i in indexes)
            else:
                def py_scores():
                    for row in valid:
                        try:
                            vec = array("f")
                            vec.frombytes(bytes(row[2]))
                            if len(vec) != len(qvec):
                                continue
                            yield str(row[0]), float(sum(map(mul, vec, qvec)))
                        except (ValueError, TypeError):
                            continue
                scored = py_scores()

            for ref, score in scored:
                if score < min_sim:
                    continue
                sequence += 1
                if keep is None:
                    all_hits.append((ref, score))
                elif len(heap) < keep:
                    heapq.heappush(heap, (score, sequence, ref))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, sequence, ref))

        try:
            cur.close()
        except Exception:
            pass
        if keep is None:
            all_hits.sort(key=lambda item: item[1], reverse=True)
            return all_hits
        return [(ref, score) for score, _seq, ref in sorted(heap, reverse=True)]

    def search(self, kind: str, qvec: array, k: int | None, min_sim: float, model: str) -> list[tuple[str, float]]:
        # Por encima del umbral no construimos una matriz con todos los BLOBs en RAM.
        threshold = max(1, int(os.getenv("HMEM_VECTOR_CACHE_MAX", "200000")))   # [PARCHE8] antes 20.000
        with self.tree.lock:
            count = int(self.tree.conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE kind = ? AND model = ?", (kind, model)
            ).fetchone()[0])
        if count > threshold:
            return self._search_streaming(kind, qvec, k, min_sim, model)

        refs, mat, dim = self._matrix(kind, model)
        if not refs or len(qvec) != dim:
            return []
        if np is not None:
            sims = mat @ np.frombuffer(qvec, dtype=np.float32)
            if k and k < len(sims):
                order = np.argpartition(-sims, k - 1)[:k]
                order = sorted(order, key=lambda i: float(sims[i]), reverse=True)
            else:
                order = np.argsort(-sims)
            return [(refs[int(i)], float(sims[int(i)])) for i in order if sims[int(i)] >= min_sim]
        scored = sorted(((sum(map(mul, v, qvec)), r) for v, r in zip(mat, refs)), reverse=True)
        if k:
            scored = scored[:k]
        return [(r, float(s)) for s, r in scored if s >= min_sim]



class EmbeddingService:
    def __init__(self, tree: Any, embedder: Any | None = None) -> None:
        self.tree = tree
        self.embedder = embedder or Embedder()
        self.model = self.embedder.model
        self.store = VectorStore(tree)
        self._qcache: OrderedDict[str, array] = OrderedDict()

    # ---------- básicos ----------
    def available(self) -> bool:
        return bool(self.embedder.available())

    def count(self) -> int:
        return self.store.count()

    def put(self, kind: str, ref: Any, vec: array) -> None:
        self.store.put(kind, ref, self.model, vec)

    def embed_query(self, text: str) -> array | None:
        text = (text or "").strip()
        if not text:
            return None
        hit = self._qcache.get(text)
        if hit is not None:
            self._qcache.move_to_end(text)
            return hit
        vecs = self.embedder.embed([text])
        if not vecs:
            return None
        self._qcache[text] = vecs[0]
        while len(self._qcache) > 32:
            self._qcache.popitem(last=False)
        return vecs[0]

    # ---------- búsquedas ----------
    def search_cells(self, question: str, k: int | None = None, include_history: bool = False) -> list[dict[str, Any]]:
        qv = self.embed_query(question)
        if qv is None:
            return []
        hits = self.store.search("cell", qv, k or int(getattr(C, "EMB_TOP_K", 15)),
                                 float(getattr(C, "EMB_MIN_SIM", 0.45)), self.model)
        if not hits:
            return []
        sims = dict(hits)
        marks = ",".join("?" * len(sims))
        with self.tree.lock:
            visibility = "" if include_history else " AND valid_until IS NULL AND superseded_by IS NULL"
            rows = self.tree.conn.execute(f"SELECT * FROM cells WHERE id IN ({marks})" + visibility, list(sims)).fetchall()
        cells = [dict(r) for r in rows]
        for c in cells:
            c["sim"] = round(sims[c["id"]], 4)
        cells.sort(key=lambda c: -c["sim"])
        return cells

    def search_episodes(self, question: str, k: int = 8) -> list[dict[str, Any]]:
        qv = self.embed_query(question)
        if qv is None:
            return []
        hits = self.store.search("episode", qv, k, float(getattr(C, "EMB_MIN_SIM", 0.45)), self.model)
        if not hits:
            return []
        sims = {int(r): s for r, s in hits}
        marks = ",".join("?" * len(sims))
        with self.tree.lock:
            rows = self.tree.conn.execute(
                f"SELECT id, ts, role, text FROM episodes WHERE id IN ({marks})", list(sims)).fetchall()
        out = [{"id": r["id"], "ts": r["ts"], "role": r["role"], "text": r["text"],
                "score": round(sims[r["id"]], 4)} for r in rows]
        out.sort(key=lambda e: -e["score"])
        return out

    def node_similarities(self, question: str) -> dict[int, float]:
        qv = self.embed_query(question)
        if qv is None:
            return {}
        hits = self.store.search("node", qv, None, -1.0, self.model)
        return {int(r): s for r, s in hits}

    def cos_nodes(self, a: int | str, b: int | str) -> float:
        """Calcula el coseno entre dos nodos. Si falta el vector, lo genera al vuelo."""
        sa, sb = str(a), str(b)
        vs = self.store.get_vectors("node", [sa, sb], self.model)
        va, vb = vs.get(sa), vs.get(sb)
        if (va is None or vb is None) and self.available():
            with self.tree.lock:
                c = self.tree.conn
                if va is None:
                    try:
                        r = c.execute("SELECT name, description FROM nodes WHERE id = ?", (int(a),)).fetchone()
                        if r:
                            txt = f"{r['name']}. {r['description'] or ''}".strip()
                            va = self.embed_query(txt)
                            if va is not None:
                                self.store.put("node", sa, self.model, va)
                    except Exception:
                        pass
                if vb is None:
                    try:
                        r = c.execute("SELECT name, description FROM nodes WHERE id = ?", (int(b),)).fetchone()
                        if r:
                            txt = f"{r['name']}. {r['description'] or ''}".strip()
                            vb = self.embed_query(txt)
                            if vb is not None:
                                self.store.put("node", sb, self.model, vb)
                    except Exception:
                        pass
        if va is None or vb is None:
            return 0.0
        return float(sum(map(mul, va, vb)))

    def cos_cells(self, a: str, b: str) -> float:
        vs = self.store.get_vectors("cell", [a, b], self.model)
        if a not in vs or b not in vs:
            return 0.0
        return float(sum(map(mul, vs[a], vs[b])))

    # ---------- colocación por vecinos ----------
    def neighbors(self, text: str, k: int = 5) -> tuple[array | None, list[tuple[dict[str, Any], float]]]:
        qv = self.embed_query(text)
        if qv is None:
            return None, []
        hits = self.store.search("cell", qv, k * 3, 0.0, self.model)
        if not hits:
            return qv, []
        sims = dict(hits)
        marks = ",".join("?" * len(sims))
        with self.tree.lock:
            rows = self.tree.conn.execute(
                f"SELECT * FROM cells WHERE archived = 0 AND valid_until IS NULL AND superseded_by IS NULL AND id IN ({marks})", list(sims)).fetchall()
        near = sorted(((dict(r), sims[r["id"]]) for r in rows), key=lambda x: -x[1])[:k]
        return qv, near

    def _ancestors(self, node_id: int) -> list[int]:
        """Ids de la cadena raíz → nodo."""
        with self.tree.lock:
            rows = self.tree.conn.execute(
                "WITH RECURSIVE up(id, parent_id, d) AS ("
                "SELECT id, parent_id, 0 FROM nodes WHERE id = ? "
                "UNION ALL SELECT n.id, n.parent_id, up.d + 1 FROM nodes n JOIN up ON n.id = up.parent_id) "
                "SELECT id FROM up ORDER BY d DESC", (node_id,)).fetchall()
        return [r[0] for r in rows]

    def pick_placement(self, near: list[tuple[dict[str, Any], float]]) -> tuple[int | None, int | None]:
        """(nodo, None) = colocar ahí; (nodo, d) = continuar la bajada con LLM desde profundidad d; (None, None) = sin pista."""
        if not near:
            return None, None
        place = float(getattr(C, "EMB_PLACE_SIM", 0.62))
        mid = float(getattr(C, "EMB_PLACE_MID", 0.45))
        share = int(getattr(C, "EMB_PLACE_SHARE", 3))
        best_cell, best = near[0]
        if best >= place:
            votes: dict[int, float] = {}
            for cell, s in near:
                if s >= place:
                    votes[cell["node_id"]] = votes.get(cell["node_id"], 0.0) + s
            return max(votes, key=votes.get), None
        if share > 0 and best >= mid:
            chain = self._ancestors(best_cell["node_id"])
            if len(chain) >= share:
                return chain[share - 1], share
        return None, None

    # ---------- tarea del ciclo de sueño ----------
    def _pending(self, limit: int) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        with self.tree.lock:
            c = self.tree.conn
            for r in c.execute(
                    "SELECT c.id, c.text FROM cells c WHERE NOT EXISTS (SELECT 1 FROM embeddings e "
                    "WHERE e.kind = 'cell' AND e.ref = c.id) LIMIT ?", (limit,)).fetchall():
                out.append(("cell", r[0], str(r[1])[:1500]))
            if len(out) < limit:
                for r in c.execute(
                        "SELECT n.id, n.name, n.description FROM nodes n WHERE NOT EXISTS (SELECT 1 FROM embeddings e "
                        "WHERE e.kind = 'node' AND e.ref = CAST(n.id AS TEXT)) LIMIT ?", (limit - len(out),)).fetchall():
                    out.append(("node", str(r[0]), f"{r[1]}. {r[2] or ''}".strip()[:500]))
            if len(out) < limit:
                try:
                    for r in c.execute(
                            "SELECT x.id, x.text FROM episodes x WHERE length(x.text) >= 15 AND NOT EXISTS ("
                            "SELECT 1 FROM embeddings e WHERE e.kind = 'episode' AND e.ref = CAST(x.id AS TEXT)) "
                            "ORDER BY x.id DESC LIMIT ?", (limit - len(out),)).fetchall():
                        out.append(("episode", str(r[0]), str(r[1])[:2000]))
                except sqlite3.Error:
                    pass
        return out

    def _cleanup(self) -> None:
        with self.tree.lock:
            c = self.tree.conn
            c.execute("DELETE FROM embeddings WHERE model != ?", (self.model,))
            c.execute("DELETE FROM embeddings WHERE kind = 'cell' AND ref NOT IN (SELECT id FROM cells)")
            c.execute("DELETE FROM embeddings WHERE kind = 'node' AND CAST(ref AS INTEGER) NOT IN (SELECT id FROM nodes)")
            try:
                c.execute("DELETE FROM embeddings WHERE kind = 'episode' "
                          "AND CAST(ref AS INTEGER) NOT IN (SELECT id FROM episodes)")
            except sqlite3.Error:
                pass
            c.commit()

    def sync(self, limit: int | None = None) -> str:
        """Genera los vectores que faltan (un lote por tick). Devuelve el informe de la tarea."""
        limit = int(limit or getattr(C, "EMB_BATCH_PER_TICK", 32))
        batch = max(1, int(getattr(C, "EMB_BATCH", 16)))
        self._cleanup()
        pend = self._pending(limit)
        if not pend:
            return "· embeddings al día"
        if not self.available():
            return "· servidor de embeddings no disponible; se reintentará"
        done = {"cell": 0, "node": 0, "episode": 0}
        for i in range(0, len(pend), batch):
            chunk = pend[i:i + batch]
            vecs = self.embedder.embed([t for _, _, t in chunk])
            if vecs is None:
                break
            self.store.put_many([(k, r, self.model, v) for (k, r, _), v in zip(chunk, vecs)])
            for k, _, _ in chunk:
                done[k] += 1
        n = sum(done.values())
        if n == 0:
            return "· el servidor de embeddings no respondió; se reintentará"
        return (f"{n} vectores generados ({done['cell']} células, {done['node']} nodos, "
                f"{done['episode']} episodios)")
