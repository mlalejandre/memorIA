# -*- coding: utf-8 -*-
# Tests del examen de cartuchos (sin red ni servidores; nunca toca body/).  [PARCHE27]
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT, ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cartucho  # noqa: E402
import config as C  # noqa: E402


class TestCartucho(unittest.TestCase):
    def test_keys_are_accent_and_thousands_insensitive(self):
        self.assertTrue(cartucho.has_keys("Murió en el hospital de Valdecilla.", ["valdecilla"]))
        self.assertTrue(cartucho.has_keys("El puerto de Maliaño", ["maliano"]))
        self.assertTrue(cartucho.has_keys("Se valoraron en 85.312.506 pesetas", ["85312506|85 312 506"]))
        self.assertTrue(cartucho.has_keys("había 50 000 habitantes", ["50000"]))
        self.assertTrue(cartucho.has_keys("Bilbao y Sevilla", ["bilbao", "sevilla"]))
        self.assertFalse(cartucho.has_keys("Bilbao", ["bilbao", "sevilla"]))
        self.assertFalse(cartucho.has_keys("cualquier cosa", []))

    def test_cause(self):
        self.assertEqual(cartucho.cause(False, False), "ingesta")
        self.assertEqual(cartucho.cause(True, False), "recuperación")
        self.assertEqual(cartucho.cause(True, True), "respuesta")

    def test_question_file_is_valid(self):
        qs = cartucho.load_questions(ROOT / "experiment" / "preguntas_santander.json")
        self.assertGreaterEqual(len(qs), 60)
        self.assertEqual(len({q["id"] for q in qs}), len(qs))
        for q in qs:
            if q["tipo"] == "trampa":
                self.assertTrue(q["trampa"])
            else:
                self.assertTrue(q["ref"] and q["claves"], q["id"])

    def test_summary_and_pairing(self):
        rows = []
        for mode, verdicts in (("no", [True, False, False]), ("auto", [True, True, False])):
            for i, v in enumerate(verdicts):
                rows.append({"id": f"q{i}", "modo": mode, "tipo": "trampa" if i == 2 else "directa", "veredicto": v,
                             "t": 1.0 + i, "causa": "respuesta", "piensa": mode == "auto"})
        s = cartucho.summarize(rows)
        self.assertEqual((s["no"]["correctas"], s["no"]["n_pos"], s["no"]["inventa"]), (1, 2, 0))
        self.assertEqual(s["auto"]["correctas"], 2)
        p = cartucho.paired(rows, "no", "auto")
        self.assertEqual((p["n"], p["mejoran"], p["empeoran"]), (3, 1, 0))

    def test_copy_reads_only(self):
        tmp = Path(tempfile.mkdtemp(prefix="hmem_cart_"))
        try:
            src = tmp / "cartuchos" / "prueba"
            src.mkdir(parents=True)
            con = sqlite3.connect(str(src / "memoria.db"))
            con.execute("CREATE TABLE cells (text TEXT)")
            con.execute("INSERT INTO cells VALUES ('hola')")
            con.commit()
            con.close()
            (src / "manifest.json").write_text(json.dumps({"id": "prueba"}), encoding="utf-8")
            before = (src / "memoria.db").read_bytes()
            with mock.patch.object(C, "CARTRIDGES_DIR", tmp / "cartuchos"):
                dst = cartucho.copy_cartridge("prueba", tmp / "copia")
            self.assertEqual((src / "memoria.db").read_bytes(), before)
            self.assertEqual(sqlite3.connect(str(dst / "memoria.db")).execute("SELECT text FROM cells").fetchone()[0], "hola")
            with self.assertRaises(FileNotFoundError):
                with mock.patch.object(C, "CARTRIDGES_DIR", tmp / "cartuchos"):
                    cartucho.copy_cartridge("no_existe", tmp / "x")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestDocumentos(unittest.TestCase):  # [PARCHE28]
    def test_prompt_and_parse(self):
        import documentos
        p = documentos.prompt("Esa misma noche arribó el crucero Canarias.", "Wikipedia: Incendio de Santander de 1941",
                              "Durante el día 18 el gobernador difundió un boletín.")
        self.assertIn("Documento: Wikipedia: Incendio de Santander de 1941", p)
        self.assertIn("día 18", p)
        self.assertIn("fecha absoluta", p)
        self.assertIn("<<<Esa misma noche", p)
        facts = documentos.parse({"hechos": [{"texto": "La noche del 18 de febrero de 1941 llegó el crucero Canarias.",
                                              "importancia": 7}, "x", {"texto": "ok hecho", "tipo": "instruccion"}]})
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[1]["tipo"], "instruccion")
        self.assertIsNone(documentos.parse("no json"))

    def test_extract_falls_back_on_error(self):
        import documentos

        class Boom:
            def ask_json(self, *a, **k):
                raise RuntimeError("caído")
        self.assertIsNone(documentos.extract(Boom(), "texto"))

    def test_default_off_and_conversations_untouched(self):
        import documentos
        self.assertFalse(documentos.enabled() and not C.DOC_SELF_CONTAINED)
        self.assertNotIn("documentos", (ROOT / "src" / "ltm" / "ingestion.py").read_text(encoding="utf-8"))

    def test_parse_texts_and_paired_variants(self):
        t = cartucho.parse_texts("a.txt=Wikipedia: A;b.txt")
        self.assertEqual([(str(p), s) for p, s in t], [("a.txt", "Wikipedia: A"), ("b.txt", "b")])
        mk = lambda i, tipo, v: {"id": i, "modo": "no", "tipo": tipo, "veredicto": v}
        a = [mk("1", "directa", False), mk("2", "directa", True), mk("3", "trampa", True)]
        b = [mk("1", "directa", True), mk("2", "directa", True), mk("3", "trampa", False)]
        pv = cartucho.paired_variants(a, b)
        self.assertEqual((pv["todas"]["mejoran"], pv["positivas"]["mejoran"], pv["trampas"]["mejoran"]), (2, 1, 1))


class TestEventRule(unittest.TestCase):  # [PARCHE29]
    def test_rule_only_when_enabled(self):
        from agente import Agent, EVENT_RULE
        with mock.patch.object(C, "ATTRIB_EVENTS", False):
            self.assertNotIn(EVENT_RULE, Agent._build_system(None, {}, {"found": False}, None))
        with mock.patch.object(C, "ATTRIB_EVENTS", True):
            self.assertIn(EVENT_RULE, Agent._build_system(None, {}, {"found": False}, None))

    def test_paired_breakdown(self):
        rows = [{"id": "1", "modo": "no", "tipo": "trampa", "veredicto": True},
                {"id": "1", "modo": "atrib", "tipo": "trampa", "veredicto": False},
                {"id": "2", "modo": "no", "tipo": "directa", "veredicto": True},
                {"id": "2", "modo": "atrib", "tipo": "directa", "veredicto": False}]
        p = cartucho.paired(rows, "no", "atrib")
        self.assertEqual((p["trampas"]["mejoran"], p["positivas"]["empeoran"]), (1, 1))

    def test_ingest_panel_guards_source(self):
        html = (ROOT / "interfaz" / "index.html").read_text(encoding="utf-8")
        self.assertIn("$('ingest-src').value = '';", html)
        self.assertIn("_lastIngestSrc", html)


class TestServerLaunch(unittest.TestCase):  # [PARCHE30]
    def test_stm_model_and_pci_order(self):
        import servidor
        with mock.patch.object(C, "REMOTE_MODEL_PATH", "/m/nail.gguf"), mock.patch.object(C, "LLAMA_SERVER_EXTRA", ""):
            cmd = servidor._launch_cmd("1", 8088, 65536, 2, "/tmp/p", "/tmp/l", "/m/qwen.gguf")
            self.assertIn("-m /m/qwen.gguf", cmd)
            self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1", cmd)
            self.assertIn("-m /m/nail.gguf", servidor._launch_cmd("0", 8087, 32768, 1, "/tmp/p", "/tmp/l"))
        # [PARCHE31] el .env manda salvo elección en la consola (BODY/servidores.json)
        env_stm = __import__("os").environ.get("REMOTE_MODEL_PATH_STM", "").strip()
        if "model" not in C.load_server_choices().get("stm", {}):
            self.assertEqual(C.REMOTE_MODEL_PATH_STM, env_stm)
