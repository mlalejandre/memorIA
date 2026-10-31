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
