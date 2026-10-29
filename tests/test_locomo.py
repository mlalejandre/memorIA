# -*- coding: utf-8 -*-
# Tests del adaptador LoCoMo (sin red, sin servidores, sin datos ajenos).  [PARCHE10]
from __future__ import annotations

import copy
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT, ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ["HMEM_NO_EMB"] = "1"

import locomo  # noqa: E402
import vectores  # noqa: E402
from largo_plazo import MemoryTree  # noqa: E402
from tests.helpers import MockEmbedder, MockLLM  # noqa: E402


class TestLocomoFormat(unittest.TestCase):
    def test_parse_dt(self):
        self.assertEqual(locomo.parse_dt("1:56 pm on 8 May, 2023"), datetime(2023, 5, 8, 13, 56))
        self.assertEqual(locomo.parse_dt("12:04 am on 19 December, 2023"), datetime(2023, 12, 19, 0, 4))
        self.assertEqual(locomo.parse_dt("12:30 pm on 1 June 2023"), datetime(2023, 6, 1, 12, 30))
        with self.assertRaises(ValueError):
            locomo.parse_dt("ayer por la tarde")

    def test_norm_evidence(self):
        self.assertEqual(locomo.norm_evidence(["D8:6; D9:17"]), {"D8:6", "D9:17"})
        self.assertEqual(locomo.norm_evidence(["D:11:26"]), {"D11:26"})
        self.assertEqual(locomo.norm_evidence(["D9:1 D4:4"]), {"D9:1", "D4:4"})
        self.assertEqual(locomo.norm_evidence(["D"]), set())

    def test_chunks_cover_every_turn_once(self):
        ch = locomo.make_chunks(locomo.MINI_SAMPLE, 3)
        dias = [d for c in ch for d in c["dia_ids"]]
        self.assertEqual(dias, ["D1:1", "D1:2", "D1:3", "D1:4", "D2:1", "D2:2", "D2:3"])
        self.assertTrue(all(c["text"].startswith("[Conversación entre Lucas y Nora") for c in ch))
        self.assertTrue(all(len({c2["session"] for c2 in [c]}) == 1 for c in ch))
        self.assertIn("comparte una foto: a photo of a lighthouse", ch[1]["text"])

    def test_cache_key_changes_with_chunk_size(self):
        self.assertEqual(locomo.cache_key("conv-1", 6), locomo.cache_key("conv-1", 6))
        self.assertNotEqual(locomo.cache_key("conv-1", 6), locomo.cache_key("conv-1", 8))
        self.assertNotEqual(locomo.cache_key("conv-1", 6, ("a",)), locomo.cache_key("conv-1", 6, ("b",)))

    def test_select_samples(self):
        data = [{"sample_id": "conv-26"}, {"sample_id": "conv-30"}]
        self.assertEqual([s["sample_id"] for s in locomo.select_samples(data, "1")], ["conv-30"])
        self.assertEqual(len(locomo.select_samples(data, "all")), 2)
        self.assertEqual([s["sample_id"] for s in locomo.select_samples(data, "conv-26")], ["conv-26"])

    def test_subsample_is_stratified_and_deterministic(self):
        qa = [{"category": 2}] * 30 + [{"category": 4}] * 60 + [{"category": 5}] * 30
        qa = [dict(q, i=i) for i, q in enumerate(qa)]
        sub = locomo.subsample(qa, 12)
        cats = [q["category"] for q in sub]
        self.assertEqual(len(sub), 12)
        self.assertEqual((cats.count(2), cats.count(4), cats.count(5)), (3, 6, 3))
        self.assertEqual(sub, locomo.subsample(qa, 12))
        self.assertEqual([q["i"] for q in sub], sorted(q["i"] for q in sub))
        self.assertEqual(locomo.subsample(qa, 0), qa)

    def test_evidence_bucket(self):
        self.assertEqual(locomo.evidence_bucket(True, True, True), "respuesta")
        self.assertEqual(locomo.evidence_bucket(False, True, True), "selección")
        self.assertEqual(locomo.evidence_bucket(False, False, True), "recuperación")
        self.assertEqual(locomo.evidence_bucket(False, False, False), "ingesta")


class TestLocomoDiagnostics(unittest.TestCase):  # [PARCHE11]
    def test_subject_of(self):
        self.assertEqual(locomo.subject_of("What did Caroline realize?", "Caroline", "Melanie"), "Caroline")
        self.assertEqual(locomo.subject_of("How long have Mel and her husband been married?", "Caroline", "Melanie"),
                         "Melanie")
        self.assertEqual(locomo.subject_of("What did Melanie tell Caroline?", "Caroline", "Melanie"), "Melanie")
        self.assertIsNone(locomo.subject_of("What is the capital of Spain?", "Caroline", "Melanie"))

    def test_phrasing_ignores_path_and_source(self):
        lines = locomo.fact_lines(
            "• [Conversaciones › Arte] Melanie pinta atardeceres. (Fuente: Conversación entre Caroline y Melanie el 1)\n"
            "• [Conversaciones] Caroline adoptó un perro. (Fuente: Conversación entre Caroline y Melanie el 2)\n"
            "• El usuario tiene un cuenco pintado a mano.\n"
            "• Se celebró un desfile.")
        self.assertEqual(len(lines), 4)
        self.assertEqual(locomo.phrasing(lines, "Melanie", "Caroline"),
                         {"sujeto": 1, "otra": 1, "usuario": 1, "sin sujeto": 1})

    def test_evidence_rank_and_coverage(self):
        cell_dia = {"a": ["D1:1", "D1:2"], "b": ["D2:1"], "c": ["D3:1"]}
        epi_dia = {"7": "D3:1"}
        self.assertEqual(locomo.evidence_rank(["c", "b", "a"], cell_dia, {"D1:2"}), 3)
        self.assertIsNone(locomo.evidence_rank(["c"], cell_dia, {"D9:9"}))
        cov = locomo.coverage({"D1:1", "D2:1", "D3:1", "D4:1"}, ["a"], [{"id": 7}], cell_dia, epi_dia)
        self.assertAlmostEqual(cov, 0.5)
        self.assertEqual(locomo.coverage(set(), ["a"], [], cell_dia, epi_dia), 0.0)


class TestLocomoComparison(unittest.TestCase):  # [PARCHE12]
    def test_sign_p(self):
        self.assertEqual(locomo.sign_p(0, 0), 1.0)
        self.assertAlmostEqual(locomo.sign_p(0, 10), 2 / 1024)
        self.assertEqual(locomo.sign_p(5, 5), 1.0)

    def test_compare_runs_pairs_by_question(self):
        base = [{"sid": "c", "q": "a", "cat": 4, "veredicto": True},
                {"sid": "c", "q": "b", "cat": 4, "veredicto": False},
                {"sid": "c", "q": "x", "cat": 5, "veredicto": True},
                {"sid": "c", "q": "y", "cat": 5, "veredicto": None},
                {"sid": "c", "q": "z", "cat": 5, "veredicto": False},
                {"sid": "c", "q": "a", "cat": 4, "veredicto": False, "ref": True}]
        new = [{"sid": "c", "q": "a", "cat": 4, "veredicto": True},
               {"sid": "c", "q": "b", "cat": 4, "veredicto": True},
               {"sid": "c", "q": "x", "cat": 5, "veredicto": False},
               {"sid": "c", "q": "y", "cat": 5, "veredicto": False},
               {"sid": "c", "q": "z", "cat": 5, "veredicto": True},
               {"sid": "c", "q": "nueva", "cat": 5, "veredicto": True}]
        cmp = locomo.compare_runs(base, new)
        self.assertEqual((cmp["pos"]["n"], cmp["pos"]["antes"], cmp["pos"]["ahora"]), (2, 1, 2))
        self.assertEqual((cmp["pos"]["mejora"], cmp["pos"]["empeora"]), (1, 0))
        self.assertEqual((cmp["neg"]["n"], cmp["neg"]["antes"], cmp["neg"]["ahora"]), (3, 1, 2))
        self.assertEqual((cmp["neg"]["mejora"], cmp["neg"]["empeora"]), (2, 1))

    def test_trap_source(self):
        self.assertEqual(locomo.trap_source("Horseback riding", ["Caroline pinta."],
                                            ["Caroline: I used to go horseback riding with my dad"]), "episódica")
        self.assertEqual(locomo.trap_source("guinea pig", ["Caroline tiene un guinea pig llamado Oscar."], []), "hechos")
        self.assertEqual(locomo.trap_source("Ferrari", ["nada"], ["nada"]), "ninguno claro")


class TestLocomoDepth(unittest.TestCase):  # [PARCHE15]
    def test_covered_and_buckets(self):
        ranks = {"D1:1": (3, None), "D2:1": (15, 8), "D3:1": (None, 40)}
        self.assertAlmostEqual(locomo.covered(ranks, 10, 6), 1 / 3)
        self.assertAlmostEqual(locomo.covered(ranks, 10, 10), 2 / 3)
        self.assertAlmostEqual(locomo.covered(ranks, 50, 50), 1.0)
        self.assertEqual(locomo.covered({}, 10, 6), 0.0)
        self.assertEqual([locomo.depth_bucket(x) for x in (1, 7, 11, 21, 51, None)],
                         ["1-6", "7-10", "11-20", "21-50", "51-200", "—"])


class TestLocomoIngestion(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hmem_locomo_"))
        self.trees = []

    def tearDown(self):
        for t in self.trees:
            try:
                t.close()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _tree(self, name):
        t = MemoryTree(self.tmp / name, MockLLM())
        t.emb = vectores.EmbeddingService(t, embedder=MockEmbedder())
        self.trees.append(t)
        return t

    def test_ingest_maps_cells_and_episodes_to_turns(self):
        tree = self._tree("a")
        st = locomo.ingest(tree, locomo.MINI_SAMPLE, 3, locomo.new_state("k", 0), log=lambda m: None)
        self.assertTrue(st["stats"]["complete"])
        self.assertEqual(st["stats"]["turns"], 7)
        self.assertEqual(sorted(st["epi_dia"].values()), ["D1:1", "D1:2", "D1:3", "D1:4", "D2:1", "D2:2", "D2:3"])
        self.assertTrue(st["cell_dia"])
        self.assertEqual(tree.episodes.count(), 7)
        with tree.lock:
            row = tree.conn.execute("SELECT fuente, valid_from FROM cells LIMIT 1").fetchone()
        self.assertIn("Conversación entre Lucas y Nora el 03/03/2024", row["fuente"])
        self.assertTrue(row["valid_from"].startswith("2024-03-03"))

    def test_ingest_resumes_after_interruption(self):
        full = locomo.ingest(self._tree("full"), locomo.MINI_SAMPLE, 2, locomo.new_state("k", 0), log=lambda m: None)

        class Corte(Exception):
            pass

        def save(st):
            if st["done"] == 2 and not st["stats"]["complete"]:
                raise Corte()

        tree = self._tree("cut")
        state = locomo.new_state("k", 0)
        with self.assertRaises(Corte):
            locomo.ingest(tree, locomo.MINI_SAMPLE, 2, state, save=save, log=lambda m: None)
        snapshot = copy.deepcopy(state)
        resumed = locomo.ingest(tree, locomo.MINI_SAMPLE, 2, snapshot, log=lambda m: None)
        self.assertTrue(resumed["stats"]["complete"])
        self.assertEqual(resumed["stats"]["turns"], full["stats"]["turns"])
        self.assertEqual(resumed["epi_dia"], full["epi_dia"])
        self.assertEqual(tree.episodes.count(), 7)
        self.assertEqual(sorted(map(sorted, resumed["cell_dia"].values())), sorted(map(sorted, full["cell_dia"].values())))

    def test_retrieve_finds_evidence_and_builds_prompt(self):
        tree = self._tree("r")
        st = locomo.ingest(tree, locomo.MINI_SAMPLE, 3, locomo.new_state("k", 0), log=lambda m: None)
        tree.emb.sync(limit=64)
        ret = locomo.retrieve(tree, "What is the name of Lucas's ferret Pimienta?", st, {"D1:1"})
        self.assertTrue(ret["has_cell"])
        self.assertTrue(ret["in_cand"])
        system = locomo.system_prompt(ret)
        self.assertIn("Fecha actual", system)
        ranks = locomo.deep_ranks(tree, "What is the name of Lucas's ferret Pimienta?", {"D1:1"}, st)  # [PARCHE15]
        self.assertIn("D1:1", ranks)
        self.assertIsNotNone(ranks["D1:1"][1])


if __name__ == "__main__":
    unittest.main()
