# -*- coding: utf-8 -*-
# Tests del adaptador LoCoMo (sin red, sin servidores, sin datos ajenos).
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


class TestLocomoDiagnostics(unittest.TestCase):
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


class TestLocomoComparison(unittest.TestCase):
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


class TestLocomoDepth(unittest.TestCase):
    def test_covered_and_buckets(self):
        ranks = {"D1:1": (3, None), "D2:1": (15, 8), "D3:1": (None, 40)}
        self.assertAlmostEqual(locomo.covered(ranks, 10, 6), 1 / 3)
        self.assertAlmostEqual(locomo.covered(ranks, 10, 10), 2 / 3)
        self.assertAlmostEqual(locomo.covered(ranks, 50, 50), 1.0)
        self.assertEqual(locomo.covered({}, 10, 6), 0.0)
        self.assertEqual([locomo.depth_bucket(x) for x in (1, 7, 11, 21, 51, None)],
                         ["1-6", "7-10", "11-20", "21-50", "51-200", "—"])


class TestLocomoDecomposition(unittest.TestCase):
    def test_oracle_subs(self):
        self.assertEqual(locomo.oracle_subs("Pimienta"), ["Pimienta"])
        self.assertEqual(locomo.oracle_subs("pottery, camping and painting"),
                         ["pottery, camping and painting", "pottery", "camping", "painting"])
        self.assertEqual(locomo.oracle_subs(""), [])

    def test_diag_pick(self):
        self.assertTrue(locomo.diag_pick("c", "q", 1, 0.0))
        self.assertTrue(locomo.diag_pick("c", "q", 3, 0.0))
        self.assertFalse(locomo.diag_pick("c", "q", 4, 0.0))
        self.assertTrue(locomo.diag_pick("c", "q", 4, 1.0))
        self.assertFalse(locomo.diag_pick("c", "q", 5, 1.0))
        picks = sum(locomo.diag_pick("c", f"q{i}", 2, 0.25) for i in range(2000))
        self.assertTrue(400 < picks < 600)

    def _rows(self, n_mh, b_full, d_full, simple_drop=0.0):
        rows = [{"cat": 1, "n_subs": 2, "t": 0.5, "cov_b": 1.0 if i < b_full else 0.5,
                 "cov_d": 1.0 if i < d_full else 0.5, "cov_o": 1.0 if i < d_full else 0.5,
                 "ctx_b": True, "ctx_d": True, "ctx_o": True} for i in range(n_mh)]
        rows += [{"cat": 4, "n_subs": 0, "t": 0.3, "cov_b": 1.0, "cov_d": 1.0 - simple_drop, "cov_o": 1.0,
                  "ctx_b": True, "ctx_d": True, "ctx_o": True} for _ in range(50)]
        return rows

    def test_decision_rule(self):
        s = locomo.decomp_summary(self._rows(200, 40, 70))
        self.assertAlmostEqual(s["cats"]["1"]["full_b"], 0.2)
        self.assertEqual((s["cats"]["1"]["mejora_d"], s["cats"]["1"]["empeora_d"]), (30, 0))
        self.assertEqual(locomo.decomp_decision(s)[0], "GO")
        self.assertEqual(locomo.decomp_decision(locomo.decomp_summary(self._rows(200, 40, 45)))[0],
                         "NO-GO (techo bajo)")
        self.assertEqual(locomo.decomp_decision(locomo.decomp_summary(self._rows(200, 40, 70, 0.05)))[0][:5],
                         "NO-GO")
        self.assertEqual(locomo.decomp_decision(locomo.decomp_summary([]))[0], "SIN DATOS")

    def test_cache_roundtrip(self):
        tmp = Path(tempfile.mkdtemp(prefix="hmem_dc_"))
        try:
            llm = MockLLM()
            c = locomo.DecompCache(tmp, "t")
            subs, _ = c.get(llm, "What did Ana and Luis cook?")
            self.assertEqual(len(subs), 2)
            c.save()
            c2 = locomo.DecompCache(tmp, "t")
            self.assertEqual(c2.get(None, "What did Ana and Luis cook?")[0], subs)
            self.assertEqual((c2.hits, c2.new), (1, 0))
            self.assertNotEqual(locomo.DecompCache(tmp, "otro").path, c.path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestLocomoRefine(unittest.TestCase):
    def _rows(self, n, b, a, r, ar, drop_a=0.0):
        rows = [{"cat": 1, "n_subs": 1, "t": 0.4, "n_ev_ses": 2, "ses_b": 3, "ses_a": 7,
                 "cov_b": 1.0 if i < b else 0.5, "cov_a": 1.0 if i < a else 0.5, "cov_r": 1.0 if i < r else 0.5,
                 "cov_ar": 1.0 if i < ar else 0.5, "cov_o": 1.0} for i in range(n)]
        rows += [{"cat": 4, "n_subs": 0, "t": 0.3, "n_ev_ses": 1, "ses_b": 3, "ses_a": 7, "cov_b": 1.0,
                  "cov_a": 1.0 - drop_a, "cov_r": 1.0, "cov_ar": 1.0 - drop_a, "cov_o": 1.0} for _ in range(50)]
        return rows

    def test_decision_prefers_simplest_within_2pp(self):
        s = locomo.refine_summary(self._rows(200, 60, 80, 83, 84))      # +10 / +11.5 / +12 pp
        self.assertEqual(locomo.refine_decision(s)[:2], ("GO", "a"))
        s = locomo.refine_summary(self._rows(200, 60, 80, 90, 91))      # div +10, 2ª +15, ambas +15.5
        self.assertEqual(locomo.refine_decision(s)[:2], ("GO", "r"))
        s = locomo.refine_summary(self._rows(200, 60, 90, 75, 91, drop_a=0.05))   # div rompe las simples
        self.assertEqual(locomo.refine_decision(s)[:2], ("GO", "r"))
        s = locomo.refine_summary(self._rows(200, 60, 64, 65, 66))
        self.assertEqual(locomo.refine_decision(s)[:2], ("NO-GO", None))
        self.assertEqual(locomo.refine_decision(locomo.refine_summary([]))[0], "SIN DATOS")
        self.assertAlmostEqual(s["cats"]["1"]["ses_a"], 7.0)

    def test_refine_cache_keys_on_context(self):
        tmp = Path(tempfile.mkdtemp(prefix="hmem_rc_"))
        try:
            c = locomo.RefineCache(tmp, "t")
            a, _ = c.get(MockLLM(), "What did Ana paint?", "- Ana pinta cerámica")
            self.assertEqual(a, ["Ana pinta cerámica"])
            c.get(MockLLM(), "What did Ana paint?", "- otra cosa")
            self.assertEqual(c.new, 2)
            c.save()
            c2 = locomo.RefineCache(tmp, "t")
            self.assertEqual(c2.get(None, "What did Ana paint?", "- Ana pinta cerámica")[0], a)
            self.assertEqual(c2.hits, 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestBenchCache(unittest.TestCase):
    def test_roundtrip_and_counters(self):
        tmp = Path(tempfile.mkdtemp(prefix="hmem_bc_"))
        try:
            bc = locomo.BenchCache(tmp / "b.sqlite")
            k = locomo.BenchCache.key("a1", "fp", 0.0, 350, "sistema", "pregunta")
            self.assertNotEqual(k, locomo.BenchCache.key("a1", "fp", 0.0, 350, "sistema2", "pregunta"))
            self.assertIsNone(bc.get_answer(k))
            bc.put_answer(k, "respuesta", 1.5)
            self.assertEqual(bc.get_answer(k), ("respuesta", 1.5))
            self.assertIsNone(bc.get_verdict("x"))
            bc.put_verdict("x", False)
            self.assertEqual(bc.get_verdict("x"), (False,))   # un False en caché no es «sin dato»
            self.assertEqual((bc.hits, bc.new), ({"a": 1, "j": 1}, {"a": 1, "j": 1}))
            bc.close()
            bc2 = locomo.BenchCache(tmp / "b.sqlite")         # persiste entre ejecuciones
            self.assertEqual(bc2.get_answer(k)[0], "respuesta")
            bc2.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fingerprint_unreachable_server(self):
        fp = locomo.server_fingerprint("http://127.0.0.1:9/v1")
        self.assertIn("error", fp)
        self.assertEqual(len(fp["id"]), 12)
        self.assertIn("slots", locomo.fp_text(fp))

    def test_judge_once_is_deterministic_and_robust(self):
        from types import SimpleNamespace
        calls = []

        def chat(messages, max_tokens=None, temperature=None):
            calls.append(temperature)
            return 'Veredicto: {"correcta": true}'
        self.assertTrue(locomo.judge_once(SimpleNamespace(chat=chat), "s", "u", "correcta"))
        self.assertEqual(calls, [0.0])

        def bad(*a, **k):
            raise RuntimeError("caído")
        self.assertIsNone(locomo.judge_once(SimpleNamespace(chat=bad), "s", "u", "correcta"))
        self.assertIsNone(locomo.judge_once(SimpleNamespace(chat=lambda *a, **k: "sin json"), "s", "u", "correcta"))


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
        ranks = locomo.deep_ranks(tree, "What is the name of Lucas's ferret Pimienta?", {"D1:1"}, st)
        self.assertIn("D1:1", ranks)
        self.assertIsNotNone(ranks["D1:1"][1])


if __name__ == "__main__":
    unittest.main()

class _R:   # informe mínimo para las pruebas
    def __init__(self) -> None:
        self.rows = []

    def add(self, suite, name, ok, value, *a, **k) -> None:
        self.rows.append((name, value))


class TestThinkingAndDeterminism(unittest.TestCase):
    def test_determinism_choice_rule(self):
        mk = lambda w, cp, same, rate: {"nombre": locomo.det_name(w, cp), "workers": w, "cache_prompt": cp,
                                        "iguales": same, "ritmo": rate}
        rows = [mk(2, True, 0.35, 40.0), mk(1, True, 0.90, 22.0), mk(2, False, 0.97, 30.0), mk(1, False, 1.0, 18.0)]
        c = locomo.determinism_choice(rows)
        self.assertEqual((c["workers"], c["cache_prompt"], c["determinista"]), (2, False, True))  # la más rápida ≥ 95 %
        c = locomo.determinism_choice(rows[:2])
        self.assertEqual((c["workers"], c["determinista"]), (1, False))        # ninguna llega: la más determinista
        self.assertIsNone(locomo.determinism_choice([])["elegida"])

    def test_probe_verdict(self):
        self.assertTrue(locomo.probe_verdict({"reasoning_chars": 300, "completion_tokens": 90}, {"completion_tokens": 5}))
        self.assertTrue(locomo.probe_verdict({"reasoning_chars": 0, "completion_tokens": 400}, {"completion_tokens": 6}))
        self.assertFalse(locomo.probe_verdict({"reasoning_chars": 0, "completion_tokens": 8}, {"completion_tokens": 6}))
        self.assertFalse(locomo.probe_verdict({"error": "URLError"}, {"completion_tokens": 6}))

    def test_thinking_probe_and_determinism_with_fake_llm(self):
        class Fake:
            def __init__(self):
                self.last_usage = {}
                self.kw = []

            def chat(self, messages, max_tokens=None, temperature=None, thinking=None, cache_prompt=None):
                self.kw.append((thinking, cache_prompt))
                self.last_usage = {"completion_tokens": 300 if thinking else 4,
                                   "reasoning_chars": 900 if thinking else 0}
                return "215" if messages[-1]["content"] == locomo.THINK_PROBE else "eco: " + messages[-1]["content"]

        f = Fake()
        pr = locomo.thinking_probe(f, 2048)
        self.assertTrue(pr["available"])
        self.assertFalse(pr["always_on"])
        self.assertEqual(pr["on"]["answer"], "215")
        tmp = Path(tempfile.mkdtemp(prefix="hmem_det_"))
        try:
            R = _R()
            items = [{"system": "s", "q": f"p{i}"} for i in range(5)]
            out = locomo.run_determinism(f, items, 0.0, R, "locomo", tmp, {"id": "x"})
            self.assertEqual(len(out["filas"]), len(locomo.DET_CONFIGS))
            self.assertTrue(all(r["iguales"] == 1.0 for r in out["filas"]))
            self.assertTrue(out["decision"]["determinista"])                   # todas deterministas: la más rápida
            self.assertIn((None, False), f.kw)                                  # cache_prompt llega por llamada
            self.assertTrue(list(tmp.glob("determinismo_*.json")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_llm_payload_overrides(self):
        import config as C
        from llm import LLM
        llm = LLM(base_url="http://127.0.0.1:9/v1", model="m")
        p = llm._payload([{"role": "user", "content": "x"}], 10, 0.0, False)
        self.assertEqual(p["chat_template_kwargs"]["enable_thinking"], bool(C.THINKING))
        self.assertEqual(p["cache_prompt"], bool(C.LLM_CACHE_PROMPT))
        p = llm._payload([{"role": "user", "content": "x"}], 10, 0.0, True, thinking=True, cache_prompt=False)
        self.assertEqual((p["chat_template_kwargs"]["enable_thinking"], p["cache_prompt"], p["stream"]),
                         (True, False, True))
        self.assertEqual(llm.last_usage, {})


class TestDeterministicGeneration(unittest.TestCase):
    def test_answer_key_compat_and_generation(self):
        k = lambda *a: locomo.answer_key("fp", 0.0, 4096, "sis", "q", *a)
        self.assertEqual(k(False, "w2cp1"), locomo.BenchCache.key("a1", "fp", 0.0, 350, "sis", "q"))          # P22
        self.assertEqual(k(True, "w2cp1"), locomo.BenchCache.key("a1", "fp", 0.0, 4096, "sis", "q", "think"))  # P23
        self.assertNotEqual(k(False, "w1cp0"), k(False, "w2cp1"))    # la base reproducible no reutiliza las antiguas
        self.assertNotEqual(k(True, "w1cp0"), k(False, "w1cp0"))
        self.assertEqual(locomo.gen_tag(1, False), "w1cp0")

    def test_think_mode_env(self):
        from unittest import mock
        for v, want in (("", "off"), ("0", "off"), ("1", "on"), ("on", "on"), ("auto", "auto"), ("AUTO", "auto")):
            with mock.patch.dict(os.environ, {"HMEM_ANSWER_THINK": v}):
                self.assertEqual(locomo.think_mode_env(), want)

