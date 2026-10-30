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
import time
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

    def test_oversized_sqlite_rejected(self):
        huge_bytes = b"SQLite format 3\x00" + b"\x00" * 100
        with mock.patch("cartuchos.MAX_IMPORT_BYTES", 50):
            res = self.mgr.import_sqlite_db(huge_bytes, "grande")
            self.assertFalse(res["ok"])
            self.assertIn("excede el límite", res["error"])

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

    def test_ipv6_host_accepted(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", "/api/status", skip_host=True)
        conn.putheader("Host", f"[::1]:{self.port}")
        conn.endheaders()
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 200)
        conn.close()

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




class TestTemporalSupersession(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hmem_temp_"))
        self.llm = MockLLM()
        self.tree = MemoryTree(self.tmp / "ltm", self.llm)
        self.tree.emb = vectores.EmbeddingService(self.tree, embedder=MockEmbedder())

    def tearDown(self):
        try:
            self.tree.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_temporal_supersession_lifecycle(self):
        # Hechos con predicado común ('trabaja') para validar FTS y supersesión en modo determinista
        old_fact = "El usuario trabaja en el Banco Santander como analista."
        new_fact = "El usuario ahora trabaja en Telefónica como ingeniero de software."
        self.llm.register(old_fact, "Datos personales", "empresa")
        self.llm.register(new_fact, "Datos personales", "empresa")

        r_old = self.tree.store(old_fact, 7)
        time.sleep(0.01)
        r_new = self.tree.store(new_fact, 8)

        # El hecho viejo debe quedar invalidado
        with self.tree.lock:
            row = self.tree.conn.execute(
                "SELECT valid_until, superseded_by, archived FROM cells WHERE id = ?",
                (r_old["id"],)
            ).fetchone()
        self.assertIsNotNone(row["valid_until"])
        self.assertEqual(row["superseded_by"], r_new["id"])
        self.assertEqual(row["archived"], 1)

        # Consulta en presente solo recupera el vigente
        res_now = self.tree.recall("¿Dónde trabaja el usuario?", reinforce=False, synthesize=False)
        self.assertIn(r_new["id"], res_now.get("candidates", []))
        self.assertNotIn(r_old["id"], res_now.get("candidates", []))

        # Consulta en pasado sí recupera el histórico
        res_hist = self.tree.recall("¿Dónde trabajaba el usuario antes?", reinforce=False, synthesize=False)
        self.assertIn(r_old["id"], res_hist.get("candidates", []))

    def test_complementary_facts_are_not_superseded(self):
        f1 = "El usuario tiene un perro llamado Thor."
        f2 = "El usuario tiene un gato llamado Misu."
        self.llm.register(f1, "Datos personales", "mascota-a")
        self.llm.register(f2, "Datos personales", "mascota-b")

        r1 = self.tree.store(f1, 7)
        self.tree.store(f2, 7)

        with self.tree.lock:
            row = self.tree.conn.execute("SELECT superseded_by FROM cells WHERE id = ?", (r1["id"],)).fetchone()
        self.assertIsNone(row["superseded_by"])


class TestRRFFusion(unittest.TestCase):
    def test_rrf_fuse_ranking_order(self):
        list_a = [{"id": "doc1", "val": "A"}, {"id": "doc2", "val": "B"}]
        list_b = [{"id": "doc2", "val": "B"}, {"id": "doc3", "val": "C"}]
        fused = vectores.fuse(list_a, list_b, limit=3, k=60)
        self.assertEqual(fused[0]["id"], "doc2")
        self.assertEqual(len(fused), 3)

    def test_rrf_fuse_empty(self):
        fused = vectores.fuse([], [], limit=5)
        self.assertEqual(fused, [])


class TestCartridgeWALCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hmem_wal_"))
        self.mgr = CartridgeManager(root=self.tmp / "cartuchos")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_import_cleans_stale_wal_files(self):
        c = self.mgr.create_cartridge("Destino", "prueba")
        cdir = self.mgr.get_cartridge_path(c["id"])
        stale_wal = cdir / "memoria.db-wal"
        stale_shm = cdir / "memoria.db-shm"
        stale_wal.write_bytes(b"residuos_wal_corruptos")
        stale_shm.write_bytes(b"residuos_shm")

        valid_db = self.tmp / "origen.db"
        conn = sqlite3.connect(str(valid_db))
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.execute("INSERT INTO t VALUES ('ok')")
        conn.commit()
        conn.close()

        res = self.mgr.import_sqlite_db(valid_db.read_bytes(), fallback_name="nuevo_origen")
        self.assertTrue(res.get("ok"))
        target_dir = self.mgr.get_cartridge_path(res["cartridge"]["id"])
        self.assertFalse((target_dir / "memoria.db-wal").exists())
        self.assertFalse((target_dir / "memoria.db-shm").exists())


class TestTentativeBandReachesPrompt(unittest.TestCase):  # [PARCHE1]
    """La banda tentativa de la compuerta debe llegar al prompt con aviso, separada de lo firme."""

    def _agent(self, results):
        from types import SimpleNamespace
        from agente import Agent
        a = Agent.__new__(Agent)
        a.state = {"ltm_task": "IDLE", "ltm_detail": ""}
        a._pause_ltm = lambda: None
        a._resume_ltm = lambda: None

        class _Tree:
            def __init__(self, r):
                self.r = r

            def recall(self, q, *args, **kwargs):
                return dict(self.r)

        a.ltm_trees = {cid: _Tree(r) for cid, r in results.items()}
        a.ltm = next(iter(a.ltm_trees.values()))
        a.cartridge_mgr = SimpleNamespace(list_cartridges=lambda: [{"id": c, "name": c.upper()} for c in results])
        return a

    def test_tentative_only(self):
        from agente import Agent, LTM_CAUTION
        a = self._agent({"x": {"found": True, "tentative": True, "answer": "• dato dudoso"}})
        res = a._query_ltm_in_process("pregunta")
        self.assertTrue(res["found"])
        self.assertTrue(res["tentative"])
        self.assertEqual(res["answer"], "")
        self.assertIn("dato dudoso", res["tentative_answer"])
        system = Agent._build_system(None, {}, res, None)
        self.assertIn(LTM_CAUTION, system)
        self.assertNotIn("Conocimiento de Memoria a Largo Plazo", system)

    def test_firm_and_tentative_are_separated(self):
        from agente import Agent, LTM_CAUTION
        a = self._agent({"x": {"found": True, "tentative": False, "answer": "• dato firme"},
                         "y": {"found": True, "tentative": True, "answer": "• dato dudoso"}})
        res = a._query_ltm_in_process("pregunta")
        self.assertFalse(res["tentative"])
        system = Agent._build_system(None, {}, res, None)
        firm_pos = system.index("dato firme")
        tent_pos = system.index("dato dudoso")
        self.assertLess(system.index("Conocimiento de Memoria a Largo Plazo"), firm_pos)
        self.assertLess(system.index(LTM_CAUTION), tent_pos)
        self.assertLess(firm_pos, system.index(LTM_CAUTION))

    def test_legacy_single_tree_dict(self):
        from agente import Agent, LTM_CAUTION
        system = Agent._build_system(None, {}, {"found": True, "tentative": True, "answer": "• viejo"}, None)
        self.assertIn(LTM_CAUTION, system)


class TestReranker(unittest.TestCase):  # [PARCHE4]
    def _rr(self, answer):
        from types import SimpleNamespace
        import rerank
        llm = SimpleNamespace(ask_json=lambda system, user, attempts=2: answer)
        return rerank.Reranker(SimpleNamespace(llm=llm))

    _CANDS = [{"id": "a", "text": "El perro del usuario se llama Thor."},
              {"id": "b", "text": "El gato del usuario se llama Misu."}]

    def test_parse_llamacpp_logits(self):
        import rerank
        data = {"results": [{"index": 1, "relevance_score": 3.2}, {"index": 0, "relevance_score": -4.1}]}
        s = rerank.parse_rerank_response(data, 2)
        self.assertGreater(s[1], 0.9)
        self.assertLess(s[0], 0.1)

    def test_parse_rejects_incomplete(self):
        import rerank
        with self.assertRaises(ValueError):
            rerank.parse_rerank_response({"results": [{"index": 0, "relevance_score": 1.0}]}, 2)

    def test_llm_judge_reorders_and_decides(self):
        with mock.patch.object(C, "RERANK_MODE", "llm"):
            out = self._rr({"responden": [2]}).rerank("¿Cómo se llama el gato?", self._CANDS)
        self.assertEqual([c["id"] for c in out["candidates"]], ["b", "a"])
        self.assertTrue(out["verdict"])
        self.assertEqual(out["n_relevant"], 1)

    def test_llm_judge_none_answers(self):
        with mock.patch.object(C, "RERANK_MODE", "llm"):
            out = self._rr({"responden": []}).rerank("¿Cómo se llama el loro?", self._CANDS)
        self.assertFalse(out["verdict"])
        self.assertEqual([c["id"] for c in out["candidates"]], ["a", "b"])

    def test_llm_judge_invalid_json_falls_back(self):
        with mock.patch.object(C, "RERANK_MODE", "llm"):
            self.assertIsNone(self._rr({}).rerank("¿Algo?", self._CANDS))

    def test_recall_policy_when_nothing_answers(self):
        tmp = Path(tempfile.mkdtemp(prefix="hmem_rr_"))
        try:
            llm = MockLLM()
            tree = MemoryTree(tmp / "ltm", llm)
            tree.emb = vectores.EmbeddingService(tree, embedder=MockEmbedder())
            llm.register("El río Ebro nace en Fontibre.", "Geografía")
            tree.store("El río Ebro nace en Fontibre.", 8)
            tree.emb.sync()

            class _NoAnswer:
                def enabled(self):
                    return True

                def rerank(self, q, cands):
                    return {"mode": "llm", "candidates": list(cands), "verdict": False, "n_relevant": 0,
                            "public": {"mode": "llm", "verdict": False}}

            tree.reranker = _NoAnswer()
            with mock.patch.object(C, "RERANK_NONE", "abstain"):
                r = tree.recall("¿Dónde nace el río Ebro?", reinforce=False, synthesize=False)
            self.assertFalse(r["found"])
            self.assertFalse(r["rerank"]["verdict"])
            with mock.patch.object(C, "RERANK_NONE", "tentative"):
                r = tree.recall("¿Dónde nace el río Ebro?", reinforce=False, synthesize=False)
            self.assertTrue(r["found"])
            self.assertTrue(r["tentative"])
            tree.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestNoDeductionRule(unittest.TestCase):  # [PARCHE6]
    def test_rule_reaches_system_prompt(self):
        from agente import Agent
        system = Agent._build_system(None, {}, {"found": False}, None)
        self.assertIn("si un dato personal no consta", system)
        self.assertIn("en vez de deducirlo", system)


class TestAttributionRule(unittest.TestCase):  # [PARCHE12]
    def test_attribution_and_relative_dates_reach_prompt(self):
        from agente import Agent
        system = Agent._build_system(None, {}, {"found": False}, None)
        self.assertIn("corrige la premisa de la pregunta", system)
        self.assertIn("a partir de la fecha de ese recuerdo", system)

    def test_episodic_block_explains_speaker(self):
        from agente import Agent
        epi = [{"ts": 1700000000.0, "role": "user", "text": "Evan: perdí mis llaves"}]
        system = Agent._build_system(None, {"recent": [], "hits": []}, {"found": False}, epi)
        self.assertIn("es de quien habla, no de su interlocutor", system)


class TestEpisodePromptK(unittest.TestCase):  # [PARCHE16]
    def _system(self, n):
        from agente import Agent
        epi = [{"ts": 1700000000.0 + i, "role": "user", "text": f"Persona{i}: episodio número {i}"} for i in range(n)]
        return Agent._build_system(None, {"recent": [], "hits": []}, {"found": False}, epi)

    def test_default_is_twenty(self):  # [PARCHE17]
        self.assertEqual(C.EPI_PROMPT_K, int(os.environ.get("EPI_PROMPT_K", "20")))
        self.assertGreaterEqual(C.EPI_LOOKUP_K, C.EPI_PROMPT_K)

    def test_lookup_never_below_prompt(self):  # [PARCHE18]
        self.assertEqual(C._episode_limits(6, 20), (22, 20))
        self.assertEqual(C._episode_limits(22, 20), (22, 20))
        self.assertEqual(C._episode_limits(30, 5), (30, 5))
        self.assertEqual(C._episode_limits(3, 0), (3, 1))

    def test_cap_of_five_still_works(self):
        with mock.patch.object(C, "EPI_PROMPT_K", 5):
            s = self._system(8)
        self.assertIn("episodio número 4", s)
        self.assertNotIn("episodio número 5", s)

    def test_cap_is_configurable(self):
        with mock.patch.object(C, "EPI_PROMPT_K", 7):
            s = self._system(8)
        self.assertIn("episodio número 6", s)
        self.assertNotIn("episodio número 7", s)


class TestDirectTopK(unittest.TestCase):  # [PARCHE13]
    def test_default_matches_config(self):
        self.assertEqual(C.LTM_DIRECT_TOP_K, int(os.environ.get("LTM_DIRECT_TOP_K", "10")))  # [PARCHE14]

    def test_top_k_limits_injected_facts(self):
        tmp = Path(tempfile.mkdtemp(prefix="hmem_topk_"))
        try:
            llm = MockLLM()
            tree = MemoryTree(tmp / "ltm", llm)
            tree.emb = vectores.EmbeddingService(tree, embedder=MockEmbedder())
            for i, lugar in enumerate(["Islandia", "Chile", "Japón", "Italia", "Hawái", "Indonesia", "México", "Kenia"]):
                txt = f"El volcán número {i} de {lugar} entró en erupción con lava basáltica."
                llm.register(txt, "Geología")
                tree.store(txt, 7)
            tree.emb.sync()
            q = "¿Qué volcán entró en erupción con lava basáltica?"
            sizes = {}
            for k in (2, 7):
                with mock.patch.object(C, "LTM_DIRECT_TOP_K", k):
                    r = tree.recall(q, reinforce=False, synthesize=False)
                self.assertTrue(r["found"])
                sizes[k] = len(r["cells"])
            self.assertEqual(sizes[2], 2)
            self.assertGreater(sizes[7], 2)
            self.assertLessEqual(sizes[7], 7)
            tree.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestDecomposition(unittest.TestCase):  # [PARCHE19]
    def test_clean_validates_llm_output(self):
        import descomponer as d
        q = "¿Qué deportes practican Ana y Luis?"
        self.assertEqual(d.clean(q, {"consultas": ["deportes de Ana", "deportes de Luis"]}),
                         ["deportes de Ana", "deportes de Luis"])
        self.assertEqual(d.clean(q, {"consultas": []}), [])
        self.assertEqual(d.clean(q, {"consultas": ["solo una consulta"]}), [])     # 1 no es descomponer
        self.assertEqual(d.clean(q, {"consultas": [q, "deportes de Ana"]}), [])    # copia de la pregunta
        self.assertEqual(d.clean(q, {"consultas": ["a", "", "deportes de Ana", "Deportes de Ana!",
                                                   "deportes de Luis", "x3", "otra", "y otra más"]}, n=3),
                         ["deportes de Ana", "deportes de Luis", "otra"])
        self.assertEqual(d.clean(q, "basura"), [])

    def test_decompose_never_raises(self):
        import descomponer as d
        from types import SimpleNamespace

        def boom(*a, **k):
            raise RuntimeError("caído")
        self.assertEqual(d.decompose(SimpleNamespace(ask_json=boom), "¿Algo?")[0], [])
        self.assertEqual(d.decompose(None, "¿Algo?")[0], [])

    def test_fuse_many_keeps_original_first_and_max_sim(self):
        import descomponer as d
        a = [{"id": "x", "sim": 0.5}, {"id": "y", "sim": 0.4}]
        b = [{"id": "z", "sim": 0.9}, {"id": "y", "sim": 0.8}]
        out = d.fuse_many([a, b], 10)
        self.assertEqual([c["id"] for c in out], ["y", "x", "z"])
        self.assertEqual(out[0]["sim"], 0.8)
        self.assertEqual(a[1]["sim"], 0.4)   # no modifica las listas de entrada
        self.assertEqual([c["id"] for c in d.fuse_many([a], 1)], ["x"])
        self.assertEqual(d.fuse_many([[], []], 5), [])

    def test_off_by_default_and_recall_unchanged_without_subqueries(self):
        self.assertEqual(C.LTM_DECOMPOSE, os.environ.get("LTM_DECOMPOSE", "0") == "1")
        tmp = Path(tempfile.mkdtemp(prefix="hmem_dec_"))
        try:
            llm = MockLLM()
            tree = MemoryTree(tmp / "ltm", llm)
            tree.emb = vectores.EmbeddingService(tree, embedder=MockEmbedder())
            facts = ["El hurón Pimienta abre la nevera.", "La cerámica de Nora se cuece en Burgos.",
                     "El torneo de ajedrez regional terminó en abril."]
            for f in facts:
                llm.register(f, "Notas")
                tree.store(f, 7)
            tree.emb.sync()
            q = "¿Qué hace el hurón Pimienta?"
            a = tree.recall(q, reinforce=False, synthesize=False)
            b = tree.recall(q, reinforce=False, synthesize=False, extra_queries=[])
            self.assertEqual(a["candidates"], b["candidates"])
            c = tree.recall(q, reinforce=False, synthesize=False,
                            extra_queries=["cerámica de Nora en Burgos", "torneo de ajedrez regional"])
            self.assertGreater(len(set(c["candidates"])), len(set(a["candidates"])) - 1)
            tree.log_episode("user", "Nora: mi cerámica se cuece en Burgos", 1.0)
            tree.log_episode("user", "Lucas: el torneo de ajedrez acabó", 2.0)
            e0 = tree.search_episodes("cerámica Burgos", 5)
            e1 = tree.search_episodes("cerámica Burgos", 5, extra_queries=["torneo ajedrez", "ajedrez acabó"])
            self.assertTrue(any("ajedrez" in e["text"] for e in e1))
            self.assertFalse(any("ajedrez" in e["text"] for e in e0))
            tree.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestRefine(unittest.TestCase):  # [PARCHE20]
    def _ep(self, i, day):
        return {"id": i, "ts": time.mktime((2024, 3, day, 12, 0, 0, 0, 0, -1)) + i, "text": f"e{i}"}

    def test_por_dia_caps_and_fills(self):
        import refinar
        items = [self._ep(i, 1) for i in range(5)] + [self._ep(10 + i, 2) for i in range(2)] + [self._ep(20, 3)]
        out = refinar.por_dia(items, 6, cap=2)
        self.assertEqual([e["id"] for e in out], [0, 1, 10, 11, 20, 2])   # 2 del día 1, luego los otros días, relleno
        self.assertEqual([e["id"] for e in refinar.por_dia(items, 3, cap=3)], [0, 1, 2])
        self.assertEqual(len(refinar.por_dia(items, 50, cap=1)), len(items))
        self.assertEqual(refinar.day_of("x"), "?")

    def test_clean_accepts_single_query(self):
        import refinar, descomponer
        self.assertEqual(refinar.clean("¿Qué investigó Ana?", {"consultas": ["Ana agencias de adopción"]}),
                         ["Ana agencias de adopción"])
        self.assertEqual(descomponer.clean("¿Qué investigó Ana?", {"consultas": ["Ana agencias de adopción"]}), [])
        self.assertEqual(refinar.clean("q", {"consultas": []}), [])

    def test_contexto_and_failure(self):
        import refinar
        from types import SimpleNamespace
        ctx = refinar.contexto(["Ana pinta", " "], [{"text": "Ana: fui a la playa"}, {"text": ""}])
        self.assertEqual(ctx, "- Ana pinta\n- Ana: fui a la playa")
        self.assertEqual(refinar.contexto([], []), "(nada)")

        def boom(*a, **k):
            raise RuntimeError("caído")
        self.assertEqual(refinar.second_pass(SimpleNamespace(ask_json=boom), "¿Algo?", ctx)[0], [])

    def test_off_by_default(self):
        self.assertEqual(C.EPI_DIVERSITY, os.environ.get("EPI_DIVERSITY", "off"))
        self.assertEqual(C.LTM_SECOND_PASS, os.environ.get("LTM_SECOND_PASS", "0") == "1")

    def test_search_episodes_diversify(self):
        tmp = Path(tempfile.mkdtemp(prefix="hmem_div_"))
        try:
            tree = MemoryTree(tmp / "ltm", MockLLM())
            tree.emb = vectores.EmbeddingService(tree, embedder=MockEmbedder())
            for i in range(6):
                tree.log_episode("user", f"Ana: hoy pinté un cuadro de cerámica número {i}", self._ep(i, 1)["ts"])
            tree.log_episode("user", "Ana: pinté cerámica en el taller", self._ep(9, 5)["ts"])
            plain = tree.search_episodes("Ana pintó cerámica", 3, diversify=False)
            div = tree.search_episodes("Ana pintó cerámica", 3, diversify=True)
            self.assertEqual(len(div), 3)
            days = lambda eps: {time.strftime("%d", time.localtime(e["ts"])) for e in eps}
            self.assertGreaterEqual(len(days(div)), len(days(plain)))
            self.assertIn("05", days(div))
            with mock.patch.object(C, "EPI_DIVERSITY", "dia"), mock.patch.object(C, "EPI_PER_DAY", 2):
                self.assertIn("05", days(tree.search_episodes("Ana pintó cerámica", 3)))
            tree.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestDirectAnswerAndFixedNow(unittest.TestCase):  # [PARCHE22]
    def test_direct_rule_only_when_enabled(self):
        from agente import Agent, DIRECT_RULE
        with mock.patch.object(C, "ANSWER_DIRECT", False):
            self.assertNotIn(DIRECT_RULE, Agent._build_system(None, {}, {"found": False}, None))
        with mock.patch.object(C, "ANSWER_DIRECT", True):
            self.assertIn(DIRECT_RULE, Agent._build_system(None, {}, {"found": False}, None))
        self.assertEqual(C.ANSWER_DIRECT, os.environ.get("ANSWER_DIRECT", "0") == "1")

    def test_fixed_now_makes_prompt_stable(self):
        from agente import Agent
        with mock.patch.dict(os.environ, {"HMEM_FIXED_NOW": "01/02/2024 10:00"}):
            s1 = Agent._build_system(None, {}, {"found": False}, None)
            self.assertIn("Fecha actual: 01/02/2024 10:00", s1)
            self.assertEqual(s1, Agent._build_system(None, {}, {"found": False}, None))

    def test_interface_imports_new_modules_from_repo(self):
        import importlib
        import interfaz
        importlib.reload(interfaz)
        self.assertIn(str(ROOT / "src"), sys.path)


class TestVectorCacheIncremental(unittest.TestCase):  # [PARCHE8]
    def test_append_without_rebuild_and_delete_rebuilds(self):
        tmp = Path(tempfile.mkdtemp(prefix="hmem_vc_"))
        try:
            tree = MemoryTree(tmp / "ltm", MockLLM())
            svc = vectores.EmbeddingService(tree, embedder=MockEmbedder())
            e = MockEmbedder()
            svc.put("cell", "a", e.embed(["perro labrador Thor"])[0])
            svc.put("cell", "b", e.embed(["gato siamés Misu"])[0])
            q = e.embed(["hurón Chispa"])[0]
            svc.store.search("cell", q, 5, -1.0, svc.model)
            self.assertEqual(svc.store.full_rebuilds, 1)
            svc.put("cell", "c", e.embed(["hurón Chispa"])[0])
            hits = svc.store.search("cell", q, 5, -1.0, svc.model)
            self.assertEqual(hits[0][0], "c")
            self.assertEqual(svc.store.incremental_appends, 1)
            self.assertEqual(svc.store.full_rebuilds, 1)
            with tree.lock:
                tree.conn.execute("DELETE FROM embeddings WHERE kind = 'cell' AND ref = 'c'")
                tree.conn.commit()
            hits = svc.store.search("cell", q, 5, -1.0, svc.model)
            self.assertNotIn("c", [h[0] for h in hits])
            self.assertEqual(svc.store.full_rebuilds, 2)
            tree.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
