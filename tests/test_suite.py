# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import http.client
import io
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ["HMEM_NO_EMB"] = "1"
os.environ["HMEM_NO_BROWSER"] = "1"

import config as C  # noqa: E402
import vectores  # noqa: E402
from cartuchos import CartridgeManager  # noqa: E402
from corto_plazo import ShortTermMemory  # noqa: E402
from largo_plazo import MemoryTree  # noqa: E402
from ltm.common import _is_nonfactual, clean_path_specs, extract_json, rrf_weight, slugify  # noqa: E402
from tests.helpers import MockEmbedder, MockLLM  # noqa: E402


class TestHierarchicalMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hmem_unit_"))
        self.llm = MockLLM()
        self.tree = MemoryTree(self.tmp / "ltm", self.llm)
        self.tree.emb = vectores.EmbeddingService(self.tree, embedder=MockEmbedder())

    def tearDown(self):
        try:
            self.tree.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _archived(self, cell_id):
        with self.tree.lock:
            return self.tree.conn.execute("SELECT archived FROM cells WHERE id = ?", (cell_id,)).fetchone()[0]

    def test_ltm_store_and_recall(self):
        self.llm.register("El río Ebro nace en Fontibre.", "Geografía")
        res_store = self.tree.store("El río Ebro nace en Fontibre.", importance=8)
        self.assertTrue(res_store["id"])
        self.tree.emb.sync()
        rec = self.tree.recall("¿Dónde nace el río Ebro?", reinforce=False, synthesize=False)
        self.assertTrue(rec["found"])
        self.assertIn("Fontibre", rec["answer"])

    def test_relevance_abstention(self):
        self.llm.register("La mitocondria produce ATP.", "Biología")
        self.tree.store("La mitocondria produce ATP.", importance=8)
        self.tree.emb.sync()
        rec = self.tree.recall("¿Cuál es la capital de Mongolia?", reinforce=False, synthesize=False)
        self.assertFalse(rec["found"])
        self.assertEqual(rec["answer"], "")

    def test_exact_duplicate_is_reinforced_not_duplicated(self):
        txt = "El monte Teide mide 3.715 metros."
        self.llm.register(txt, "Geografía")
        first = self.tree.store(txt, importance=6)
        second = self.tree.store(txt, importance=6)
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["id"], second["id"])

    def test_decay_archives_trivia_and_keeps_vital(self):
        low = "El usuario anotó un dato trivial sin importancia."
        high = "Directiva vital: proteger siempre las claves privadas."
        self.llm.register(low, "Notas")
        self.llm.register(high, "Notas")
        id_low = self.tree.store(low, importance=1)["id"]
        id_high = self.tree.store(high, importance=10)["id"]
        self.tree.advance(400)
        self.tree.task_decay()
        self.assertEqual(self._archived(id_low), 1)
        self.assertEqual(self._archived(id_high), 0)

    def test_archived_cell_is_restored_when_recalled(self):
        txt = "La fumarola principal del volcán Fuji emite vapor."
        self.llm.register(txt, "Geología")
        cid = self.tree.store(txt, importance=1)["id"]
        self.tree.advance(400)
        self.tree.task_decay()
        self.assertEqual(self._archived(cid), 1)
        self.tree.recall("fumarola volcán Fuji vapor", reinforce=True, synthesize=False)
        self.assertEqual(self._archived(cid), 0)


class TestPureFunctions(unittest.TestCase):
    def test_nonfactual_filter(self):
        self.assertTrue(_is_nonfactual("¿Cuál es la capital de Francia?"))
        self.assertTrue(_is_nonfactual("Hola"))
        self.assertFalse(_is_nonfactual("Mi perro se llama Thor y tiene cuatro años."))

    def test_rrf_weights_follow_config(self):
        self.assertEqual(rrf_weight("embeddings (coseno)"), C.RRF_W_EMB)
        self.assertEqual(rrf_weight("exacta"), C.RRF_W_EXACT)
        self.assertEqual(rrf_weight("canal-desconocido"), 1.0)

    def test_extract_json(self):
        self.assertEqual(extract_json('texto {"a": 1} fin'), {"a": 1})
        with self.assertRaises(ValueError):
            extract_json("sin json aquí")

    def test_clean_path_specs_drops_generic_nodes(self):
        out = clean_path_specs([{"name": "general"}, {"name": "Riñón", "description": "Órgano"}])
        self.assertEqual([s["name"] for s in out], ["Riñón"])

    def test_slugify(self):
        self.assertEqual(slugify("Fisiología Médica"), "fisiologia-medica")

    def test_temporal_cues(self):
        f = MemoryTree._looks_like_temporal_update
        self.assertTrue(f("El usuario vive en Valencia.", "El usuario se ha mudado a Tokio."))
        self.assertFalse(f("El usuario tiene un perro llamado Thor.", "El usuario tiene un gato llamado Misu."))

    def test_history_detection(self):
        w = MemoryTree._wants_history
        self.assertTrue(w("¿Dónde vivía antes?"))
        self.assertFalse(w("¿Dónde vive?"))


class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hmem_stm_"))
        self.llm = MockLLM()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lifecycle_and_recall(self):
        stm = ShortTermMemory(self.llm, path=self.tmp / "a", max_tokens=1000)
        stm.add("user", "Hola, me llamo Marcos y vivo en Madrid.")
        stm.add("assistant", "Encantado Marcos.")
        self.assertEqual(len(stm.snapshot()), 2)
        hit = stm.recall("¿Dónde vivo?")
        self.assertTrue(any("Madrid" in h["text"] for h in hit["recent"]))
        stm.stop()

    def test_compaction_reduces_tokens_and_keeps_recent(self):
        stm = ShortTermMemory(self.llm, path=self.tmp / "b", max_tokens=1500, trigger_ratio=0.8, target_ratio=0.5)
        for i in range(40):
            stm.add("user" if i % 2 == 0 else "assistant",
                    f"Mensaje {i}: seguimiento del proyecto Atlas con métricas {i * 37} y notas de revisión semanal.")
        before = stm.total_tokens()
        last_before = [e["text"] for e in stm.snapshot()][-stm.keep_recent:]
        stm.compact(force=True)
        after = stm.total_tokens()
        last_after = [e["text"] for e in stm.snapshot()][-stm.keep_recent:]
        self.assertLessEqual(after, before * 0.7)
        self.assertEqual(last_before, last_after)
        stm.stop()

    def test_corrupt_session_file_is_preserved(self):
        d = self.tmp / "c"
        d.mkdir()
        (d / "sesion.json").write_text("{esto no es json", encoding="utf-8")
        ShortTermMemory(self.llm, path=d)
        self.assertTrue(any(p.name.startswith("sesion.json.corrupt") for p in d.iterdir()))


class TestCartridges(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hmem_cart_"))
        self.llm = MockLLM()
        body = self.tmp / "body"
        self._patches = [
            mock.patch.object(C, "BODY", body),
            mock.patch.object(C, "LTM_DIR", body / "largo_plazo"),
            mock.patch.object(C, "STM_DIR", body / "corto_plazo"),
        ]
        for p in self._patches:
            p.start()
        self.mgr = CartridgeManager(root=self.tmp / "cartuchos")
        self.trees = []

    def tearDown(self):
        for t in self.trees:
            try:
                t.close()
            except Exception:
                pass
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _tree(self, cid):
        t = MemoryTree(self.mgr.get_cartridge_path(cid), self.llm)
        self.trees.append(t)
        return t

    def test_path_traversal_rejected(self):
        for bad in ("../../etc/passwd", "..", "a/b", "", "x" * 80):
            with self.assertRaises(ValueError):
                self.mgr.get_cartridge_path(bad)

    def test_hostile_names_stay_inside_root(self):
        c = self.mgr.create_cartridge("../../Evil/../Nombre", "x")
        self.assertEqual((self.mgr.root / c["id"]).resolve().parent, self.mgr.root.resolve())

    def test_primary_cannot_be_deleted(self):
        self.assertFalse(self.mgr.delete_cartridge(self.mgr.get_primary_id())["ok"])

    def test_isolation_between_cartridges(self):
        a = self.mgr.create_cartridge("Biología", "x")
        b = self.mgr.create_cartridge("Astro", "y")
        ta, tb = self._tree(a["id"]), self._tree(b["id"])
        self.llm.register("Las mitocondrias generan ATP.", "Biología")
        self.llm.register("Las estrellas de neutrones son densas.", "Astronomía")
        ta.store("Las mitocondrias generan ATP.", 8)
        tb.store("Las estrellas de neutrones son densas.", 8)
        self.assertEqual(len(ta._exact_search("neutrones")), 0)
        self.assertEqual(len(tb._exact_search("mitocondrias")), 0)

    def test_export_import_roundtrip(self):
        c = self.mgr.create_cartridge("Biología", "x")
        t = self._tree(c["id"])
        txt = "Los ribosomas sintetizan proteínas."
        self.llm.register(txt, "Biología")
        t.store(txt, 8)
        t.close()
        arc = self.mgr.export_cartridge_tar(c["id"])
        res = self.mgr.import_cartridge_tar(arc, "copia")
        self.assertTrue(res["ok"], res)
        t2 = self._tree(res["cartridge"]["id"])
        self.assertEqual(t2.counts()["alive"], 1)

    def test_zip_slip_rejected(self):
        c = self.mgr.create_cartridge("Base", "x")
        self._tree(c["id"]).close()
        evil_name = f"hmem_evil_{os.getpid()}.txt"
        evil = Path(tempfile.gettempdir()) / evil_name
        evil.unlink(missing_ok=True)
        arc = self.tmp / "zipslip.cartridge"
        with tarfile.open(arc, "w:gz") as tar:
            info = tarfile.TarInfo("../" + evil_name)
            payload = b"pwned"
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            tar.add(self.mgr.get_cartridge_path(c["id"]) / "memoria.db", arcname="memoria.db")
        res = self.mgr.import_cartridge_tar(arc, "zipslip")
        wrote = evil.exists()
        evil.unlink(missing_ok=True)
        self.assertFalse(res.get("ok"))
        self.assertFalse(wrote)

    def test_garbage_rejected(self):
        self.assertFalse(self.mgr.import_cartridge_b64("basura.cartridge", "AAAA" * 50).get("ok"))

    def test_foreign_db_with_trigger_rejected(self):
        db = self.tmp / "evil.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (x)")
        conn.execute("CREATE TRIGGER evil AFTER INSERT ON t BEGIN SELECT 1; END")
        conn.commit()
        conn.close()
        b64 = base64.b64encode(db.read_bytes()).decode()
        self.assertFalse(self.mgr.import_cartridge_b64("evil.db", b64).get("ok"))


class _StubAgent:
    cartridge_mgr = None

    def status(self):
        return {"ok": True}

    def chat(self, text):
        return {"answer": "ok"}


class TestWebSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from interfaz import frontend
        cls.srv = frontend.ThreadedHTTPServer(("127.0.0.1", 0), frontend.Handler)
        cls.srv.agent = _StubAgent()
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def _req(self, method, path, host=None, origin=None, content_length=None, body=b""):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", host or f"127.0.0.1:{self.port}")
        if origin is not None:
            conn.putheader("Origin", origin)
        if method == "POST":
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(content_length if content_length is not None else len(body)))
        conn.endheaders(body if content_length is None else b"")
        resp = conn.getresponse()
        resp.read()
        status = resp.status
        conn.close()
        return status

    def test_local_get_ok(self):
        self.assertEqual(self._req("GET", "/api/status"), 200)

    def test_foreign_host_rejected(self):
        self.assertEqual(self._req("GET", "/api/status", host="evil.com"), 403)
        self.assertEqual(self._req("GET", "/api/status", host="127.0.0.1.evil.com"), 403)

    def test_foreign_origin_rejected(self):
        self.assertEqual(self._req("POST", "/api/chat", origin="http://evil.com", body=b"{}"), 403)
        self.assertEqual(self._req("POST", "/api/chat", origin="null", body=b"{}"), 403)

    def test_oversized_body_rejected(self):
        self.assertEqual(self._req("POST", "/api/chat", content_length=30 * 1024 * 1024), 400)


if __name__ == "__main__":
    unittest.main()
