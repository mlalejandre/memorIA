# -*- coding: utf-8 -*-
# Tests del selector de modelos y GPUs (sin red ni SSH: un servidor remoto simulado; nunca toca body/).  [PARCHE31]
from __future__ import annotations

import re
import shlex
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config as C  # noqa: E402
import modelos  # noqa: E402
import servidor  # noqa: E402

NAIL = "/mnt/IA/models/nail-35b/Nail-Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
QWEN = "/mnt/IA/models/qwen3.8-27b-gsq-rco-iq3_s/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
BGE = "/mnt/IA/models/embeddings/bge-m3-Q8_0.gguf"
BIN = "/home/u/llama.cpp/build/bin/llama-server"
BIN75 = "/home/u/llama.cpp/build-sm75/bin/llama-server"


def llm_argv(model, port, ctx, np_):
    return [BIN, "-m", model, "--alias", "nail-35b", "--host", "0.0.0.0", "--port", str(port), "-ngl", "all",
            "--split-mode", "none", "--flash-attn", "on", "-c", str(ctx), "-np", str(np_), "-ctk", "q8_0",
            "-ctv", "q8_0", "--threads", "8"]


class FakeRemote:
    """Simula el host de las GPU: procesos por puerto, nvidia-smi, kill, nohup, tail."""

    def __init__(self):
        self.gpus = [("0", "00000000:01:00.0", "NVIDIA GeForce RTX 4090", 24564),
                     ("1", "00000000:08:00.0", "NVIDIA GeForce RTX 3090", 24576),
                     ("2", "00000000:09:00.0", "NVIDIA GeForce RTX 2060", 6144)]
        self.procs = {}     # pid -> dict(port, argv, env, bus, mb, alive, ready_in)
        self.next_pid = 5000
        self.cmds = []
        self.files = [(19_000_000_000, NAIL), (11_800_000_000, QWEN), (635_000_000, BGE),
                      (900_000_000, "/mnt/IA/models/qwen3.8-27b-gsq-rco-iq3_s/mmproj-F16.gguf"),
                      (5_000_000_000, "/mnt/IA/models/big/Big-00001-of-00002.gguf"),
                      (4_000_000_000, "/mnt/IA/models/big/Big-00002-of-00002.gguf")]
        self.add(8087, llm_argv(NAIL, 8087, 32768, 1), {"CUDA_VISIBLE_DEVICES": "0"}, "01:00.0", 21000)
        self.add(8088, llm_argv(NAIL, 8088, 81920, 2), {"CUDA_VISIBLE_DEVICES": "1"}, "08:00.0", 22000)
        self.add(8089, [BIN, "-m", BGE, "--alias", "bge-m3", "--port", "8089", "--embeddings", "-c", "8192"],
                 {"CUDA_VISIBLE_DEVICES": "1"}, "08:00.0", 1500)
        self.fail_models = set()
        # [PARCHE32] tu compilación: 3090 y 4090 sí, 2060 no
        self.caches = {BIN: ["CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=86;89", "GGML_NATIVE:BOOL=OFF"],
                       BIN75: ["CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=75", "GGML_NATIVE:BOOL=OFF"]}

    def add(self, port, argv, env, bus, mb, ready_in=0):
        pid = str(self.next_pid)
        self.next_pid += 1
        self.procs[pid] = {"port": port, "argv": argv, "env": env, "bus": bus, "mb": mb, "alive": True,
                           "ready_in": ready_in}
        return pid

    def listener(self, port):
        return next((pid for pid, p in self.procs.items() if p["alive"] and p["port"] == port), None)

    def health(self, port):
        pid = self.listener(port)
        if not pid:
            return False
        p = self.procs[pid]
        if p["ready_in"] > 0:
            p["ready_in"] -= 1
            return False
        return True

    def _gpu_of(self, env):
        idx = env.get("CUDA_VISIBLE_DEVICES", "0")
        return next(g for g in self.gpus if g[0] == idx)[1].split(":", 1)[1].lower()

    def run(self, cmd, timeout=15):
        self.cmds.append(cmd)
        out = ""
        if cmd.startswith("bash -c "):
            cmd = shlex.split(cmd)[2]
        if "##GPUS" in cmd or "##MODELS" in cmd:
            lines = []
            if "##GPUS" in cmd:
                lines.append("##GPUS")
                for idx, bus, name, total in self.gpus:
                    used = sum(p["mb"] for p in self.procs.values() if p["alive"] and p["bus"] == bus.split(":", 1)[1].lower())
                    cap = {"0": "8.9", "1": "8.6", "2": "7.5"}[idx]
                    lines.append(f"c6, {idx}, {bus}, {name}, {used}, {total}, {cap}")
                lines.append("##APPS")
                for pid, p in self.procs.items():
                    if p["alive"]:
                        lines.append(f"{pid}, 00000000:{p['bus'].upper()}, {p['mb']}")
                for b in re.findall(r"##BIN (\S+)'", cmd):
                    lines += [f"##BIN {b}", f"path={b}"] + [f"cache={c}" for c in self.caches.get(b, [])]
                for port in re.findall(r"##PORT (\d+)", cmd):
                    pid = self.listener(int(port))
                    lines += [f"##PORT {port}", f"pid={pid or ''}"]
                    if pid:
                        p = self.procs[pid]
                        lines.append(f"exe={p['argv'][0]}")
                        lines += [f"cache={c}" for c in self.caches.get(p["argv"][0], [])]
                        lines += [f"arg={a}" for a in p["argv"]] + ["arg="]
                        lines += [f"env={k}={v}" for k, v in p["env"].items()]
            lines.append("##MODELS")
            lines += [f"{s}\t{f}" for s, f in self.files]
            lines.append("##END")
            out = "\n".join(lines) + "\n"
        elif "echo STOPPED" in cmd:
            pid = re.search(r"pid=(\d+)", cmd).group(1)
            if pid in self.procs and self.procs[pid]["alive"]:
                self.procs[pid]["alive"] = False
                out = "STOPPED\n"
            else:
                out = "GONE\n"
        elif " nohup " in f" {cmd}":
            line = cmd.split(" > ")[0].split(" >> ")[0]
            toks = shlex.split(line)
            env, i = {}, 0
            if toks[0] == "env":
                i = 1
                while toks[i] == "-u":
                    i += 2
            while "=" in toks[i] and not toks[i].startswith("-"):
                k, v = toks[i].split("=", 1)
                env[k] = v
                i += 1
            assert toks[i] == "nohup", toks
            argv = toks[i + 1:]
            model = argv[argv.index("-m") + 1]
            port = int(argv[argv.index("--port") + 1])
            mb = 1500 if "--embeddings" in argv else 12000 if model == QWEN else 21000
            pid = self.add(port, argv, env, self._gpu_of(env), mb, ready_in=2)
            if model in self.fail_models:
                self.procs[pid]["alive"] = False
            out = f"pid={pid}\n"
        elif "echo VIVO" in cmd:
            pid = re.search(r"kill -0 (\d+)", cmd).group(1)
            out = "VIVO\n" if self.procs.get(pid, {}).get("alive") else "MUERTO\n"
        elif cmd.startswith("tail"):
            out = "llama_model_load: error: out of memory\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


class TestModelos(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = {k: getattr(C, k) for k in ("REMOTE_MODEL_PATH", "REMOTE_MODEL_PATH_STM", "REMOTE_MODEL_PATH_LTM",
                                                 "GPU0_INDEX", "GPU1_INDEX", "GPU2_INDEX", "SLOT_CTX_GPU1",
                                                 "REMOTE_CTX_GPU0", "PARALLEL_SLOTS_GPU1", "EMB_MODEL_PATH",
                                                 "REMOTE_MODELS_DIR", "SERVERS_CHOICE_FILE", "ENV_SERVER_DEFAULTS",
                                                 "GPU_SERVER_BINS", "LLAMA_SERVER_BIN",
                                                 "SERVER_PORT_GPU0", "SERVER_PORT_GPU1", "SERVER_PORT_GPU2")}
        self.flags = {k: getattr(servidor, k) for k in ("_started_gpu0", "_started_gpu1", "_started_gpu2")}
        C.REMOTE_MODEL_PATH, C.REMOTE_MODEL_PATH_STM, C.REMOTE_MODEL_PATH_LTM = NAIL, "", ""
        C.GPU0_INDEX, C.GPU1_INDEX, C.GPU2_INDEX = "0", "1", "1"
        C.SLOT_CTX_GPU1, C.PARALLEL_SLOTS_GPU1, C.REMOTE_CTX_GPU0 = 40960, 2, 32768
        C.SERVER_PORT_GPU0, C.SERVER_PORT_GPU1, C.SERVER_PORT_GPU2 = 8087, 8088, 8089
        C.EMB_MODEL_PATH, C.REMOTE_MODELS_DIR = BGE, ""
        C.GPU_SERVER_BINS, C.LLAMA_SERVER_BIN = {}, BIN
        C.SERVERS_CHOICE_FILE = Path(self.tmp.name) / "servidores.json"
        C.ENV_SERVER_DEFAULTS = {"ltm": {"model": NAIL, "gpu": "0", "ctx": 32768},
                                 "stm": {"model": NAIL, "gpu": "1", "ctx": 81920},
                                 "emb": {"model": BGE, "gpu": "1", "ctx": 8192}}
        self.r = FakeRemote()
        self.props = mock.patch.object(modelos, "props", lambda port: {})
        self.props.start()

    def tearDown(self):
        self.props.stop()
        for k, v in self.saved.items():
            setattr(C, k, v)
        for k, v in self.flags.items():
            setattr(servidor, k, v)
        self.tmp.cleanup()

    def inv(self):
        return modelos.inventory(self.r.run, self.r.health)

    def events(self, role, **kw):
        return list(modelos.switch(role, run=self.r.run, health=self.r.health, sleep=lambda s: None, **kw))

    # ---------------------------------------------------------------- inventario
    def test_inventory_reads_gpus_servers_and_models(self):
        inv = self.inv()
        self.assertNotIn("error", inv)
        self.assertEqual([g["index"] for g in inv["gpus"]], ["0", "1", "2"])
        stm, emb = inv["servers"]["stm"], inv["servers"]["emb"]
        self.assertEqual((stm["model"], stm["gpu"], stm["ctx"], stm["slots"]), (NAIL, "1", 81920, 2))
        self.assertEqual(emb["gpu"], "1")          # los embeddings en la 3090: lo que vimos en tu servidor
        self.assertIn("EMB", inv["gpus"][1]["servers"])
        names = {m["name"]: m for m in inv["models"]}
        self.assertIn("qwen3.8-27b-gsq-rco-iq3_s/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf", names)
        self.assertFalse(any("mmproj" in n for n in names))
        self.assertEqual(names["big/Big-00001-of-00002.gguf"]["parts"], 2)
        self.assertEqual(names["big/Big-00001-of-00002.gguf"]["gb"], 9.0)
        self.assertEqual(names["embeddings/bge-m3-Q8_0.gguf"]["kind"], "emb")
        self.assertEqual(inv["dirs"], ["/mnt/IA/models"])

    def test_inventory_reports_ssh_failure(self):
        bad = lambda cmd, timeout=15: SimpleNamespace(returncode=255, stdout="", stderr="Permission denied (publickey).")
        inv = modelos.inventory(bad, lambda p: False)
        self.assertIn("error", inv)
        self.assertIn("ssh-copy-id", inv["hint"])

    def test_script_does_not_match_itself(self):
        s = modelos._script([8088], ["/m"])
        self.assertIn("[-]-port 8088( |$)", s)
        self.assertIsNone(re.search(r"--port 8088( |$)", s))

    # ---------------------------------------------------------------- plan
    def test_plan_keeps_current_command_and_changes_only_model_ctx_gpu(self):
        p = modelos.plan("stm", self.inv(), QWEN, None, 65536)
        argv = p["argv"]
        self.assertEqual(argv[argv.index("-m") + 1], QWEN)
        self.assertEqual(argv[argv.index("-c") + 1], "65536")
        for keep in ("--alias", "nail-35b", "-ctk", "q8_0", "-np"):
            self.assertIn(keep, argv)
        self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1", p["cmd"])
        self.assertIn("env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=1 nohup", p["rollback"])
        self.assertIn(NAIL, p["rollback"])

    def test_plan_validations(self):
        inv = self.inv()
        with self.assertRaises(ValueError):
            modelos.plan("stm", inv, "/tmp/inventado.gguf")
        with self.assertRaises(ValueError):
            modelos.plan("stm", inv, BGE)                     # embeddings no sirven para chatear
        with self.assertRaises(ValueError):
            modelos.plan("stm", inv, NAIL, "7")               # no hay GPU 7
        with self.assertRaises(ValueError):
            modelos.plan("stm", inv, NAIL, "1", 81920)        # ya está así
        with self.assertRaises(ValueError):
            modelos.plan("emb", inv, QWEN, "2")               # el modelo de embeddings no se cambia
        with self.assertRaises(ValueError):
            modelos.plan("stm", inv, QWEN, None, 100)         # contexto absurdo
        with self.assertRaises(modelos.NeedsForce):
            modelos.plan("ltm", inv, NAIL, "1")               # 19 GB no caben junto al STM en la 3090
        self.assertEqual(modelos.plan("ltm", inv, NAIL, "1", force=True)["gpu"], "1")

    def test_plan_counts_vram_freed_by_the_server_it_replaces(self):
        inv = self.inv()   # la 3090 tiene 23,5 GB en uso, 22 de ellos del STM actual
        p = modelos.plan("stm", inv, QWEN)
        self.assertGreater(p["free_mb"], p["need_mb"])

    def test_plan_emb_moves_gpu_only(self):
        p = modelos.plan("emb", self.inv(), None, "0")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", p["cmd"])
        self.assertIn(BGE, p["cmd"])
        self.assertIn("--embeddings", p["argv"])

    def test_resolve_model(self):
        models = self.inv()["models"]
        self.assertEqual(modelos.resolve_model("qwen3.8", models), QWEN)
        self.assertEqual(modelos.resolve_model(QWEN, models), QWEN)
        with self.assertRaises(ValueError):
            modelos.resolve_model("gguf", models)     # ambiguo
        with self.assertRaises(ValueError):
            modelos.resolve_model("llama-70b", models)

    # ---------------------------------------------------------------- cambio
    def test_switch_success_saves_choice_and_updates_config(self):
        evs = self.events("stm", model=QWEN, ctx=65536)
        self.assertEqual(evs[-1]["event"], "done", evs)
        new = self.r.listener(8088)
        self.assertEqual(self.r.procs[new]["argv"][self.r.procs[new]["argv"].index("-m") + 1], QWEN)
        self.assertEqual(self.r.procs[new]["env"], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(sum(1 for p in self.r.procs.values() if p["alive"] and p["port"] == 8088), 1)
        self.assertEqual(C.load_server_choices(), {"stm": {"model": QWEN, "ctx": 65536}})
        self.assertEqual((C.REMOTE_MODEL_PATH_STM, C.SLOT_CTX_GPU1), (QWEN, 32768))
        self.assertFalse(servidor._started_gpu1)   # el anterior no era de esta sesión: el nuevo tampoco se para al salir

    def test_switch_back_to_env_removes_choice(self):
        self.events("stm", model=QWEN, ctx=65536)
        evs = self.events("stm", model=NAIL, ctx=81920)
        self.assertEqual(evs[-1]["event"], "done", evs)
        self.assertEqual(C.load_server_choices(), {})

    def test_switch_failure_rolls_back_to_exact_previous_command(self):
        before = self.r.procs[self.r.listener(8088)]
        self.r.fail_models.add(QWEN)
        evs = self.events("stm", model=QWEN)
        self.assertEqual(evs[-1]["event"], "error")
        self.assertTrue(evs[-1]["rolled_back"])
        now = self.r.procs[self.r.listener(8088)]
        self.assertEqual(now["argv"], before["argv"])
        self.assertEqual(now["env"], before["env"])     # sin CUDA_DEVICE_ORDER, como estaba
        self.assertEqual(C.load_server_choices(), {})   # nada guardado

    def test_switch_without_save(self):
        evs = self.events("emb", gpu="0", save=False)
        self.assertEqual(evs[-1]["event"], "done", evs)
        self.assertEqual(self.r.procs[self.r.listener(8089)]["bus"], "01:00.0")
        self.assertFalse(C.SERVERS_CHOICE_FILE.exists())

    def test_switch_starts_stopped_server(self):
        self.r.procs[self.r.listener(8087)]["alive"] = False
        evs = self.events("ltm", model=QWEN, gpu="0")
        self.assertEqual(evs[-1]["event"], "done", evs)
        self.assertTrue(servidor._started_gpu0)     # lo lanzó esta sesión
        self.assertEqual(C.load_server_choices(), {"ltm": {"model": QWEN}})

    def test_needs_force_event(self):
        evs = self.events("ltm", model=NAIL, gpu="1")
        self.assertTrue(evs[-1].get("need_force"))
        self.assertTrue(self.r.listener(8087))      # no se ha parado nada

    def test_guarded_refuses_when_agent_busy(self):
        busy = SimpleNamespace(acquire=lambda timeout: False, release=lambda: None)
        with mock.patch.object(modelos, "switch", lambda *a, **k: iter([{"event": "done", "text": "x"}])):
            evs = list(modelos.switch_guarded(SimpleNamespace(_chat_lock=busy), "stm"))
            self.assertEqual(evs[-1]["event"], "error")
            agent = SimpleNamespace(_chat_lock=threading.Lock(), ltm_lock=threading.Lock(), logs=[])
            agent.log = agent.logs.append
            evs = list(modelos.switch_guarded(agent, "stm"))
            self.assertEqual(evs[-1]["event"], "done")
            self.assertFalse(agent._chat_lock.locked() or agent.ltm_lock.locked())
            self.assertTrue(agent.logs)

    # ---------------------------------------------------------------- [PARCHE32] arquitecturas CUDA
    def test_arch_ok(self):
        b = {"archs": ["86", "89"]}
        self.assertTrue(modelos.arch_ok("8.6", b))
        self.assertTrue(modelos.arch_ok("8.9", b))
        self.assertFalse(modelos.arch_ok("7.5", b))
        self.assertTrue(modelos.arch_ok("7.5", {"archs": ["75"]}))
        self.assertTrue(modelos.arch_ok("8.9", {"archs": ["75"]}))          # PTX: sirve en GPUs más nuevas
        self.assertFalse(modelos.arch_ok("8.6", {"archs": ["75-real"]}))
        self.assertIsNone(modelos.arch_ok("7.5", {"archs": ["native"]}))
        self.assertIsNone(modelos.arch_ok("7.5", {"archs": None}))
        self.assertIsNone(modelos.arch_ok(None, b))

    def test_inventory_marks_2060_incompatible(self):
        inv = self.inv()
        self.assertEqual([g["cap"] for g in inv["gpus"]], ["8.9", "8.6", "7.5"])
        emb = inv["servers"]["emb"]
        self.assertEqual(emb["gpu_ok"], {"0": True, "1": True, "2": False})
        self.assertEqual(inv["gpus"][2]["incompatible"], ["STM", "LTM", "EMB"])
        self.assertTrue(any("scripts/modelos.py --compilar-gpu 2" in w for w in inv["warnings"]))

    def test_plan_refuses_incompatible_gpu_even_forced(self):
        with self.assertRaises(ValueError) as cm:
            modelos.plan("emb", self.inv(), None, "2", force=True)
        self.assertIn("86;89", str(cm.exception))
        self.assertNotIsInstance(cm.exception, modelos.NeedsForce)

    def test_gpu_server_bin_used_for_that_gpu_only(self):
        C.GPU_SERVER_BINS = {"2": BIN75}
        inv = self.inv()
        self.assertTrue(inv["servers"]["emb"]["gpu_ok"]["2"])
        p = modelos.plan("emb", inv, None, "2")
        self.assertEqual(p["argv"][0], BIN75)
        self.assertIn(BIN, p["rollback"])                      # para volver, el binario de antes
        evs = self.events("emb", gpu="2", save=False)
        self.assertEqual(evs[-1]["event"], "done", evs)
        back = modelos.plan("emb", self.inv(), None, "1")      # de la 2060 a la 3090: vuelve al binario del .env
        self.assertEqual(back["argv"][0], BIN)

    def test_resolve_prefers_exact_ending(self):
        mtp = "/mnt/IA/models/qwen3.8-27b-gsq-rco-iq3_s-mtp/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
        self.r.files.append((12_100_000_000, mtp))
        models = self.inv()["models"]
        self.assertEqual(modelos.resolve_model("GSQ-RCO-IQ3_S", models), QWEN)
        self.assertEqual(modelos.resolve_model("iq3_s-mtp", models), mtp)
        self.assertEqual(modelos.resolve_model("Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf", models), mtp)

    def test_build_script_keeps_your_build(self):
        sh, out, log = modelos.build_script("/home/u/llama.cpp/build", "75")
        self.assertEqual(out, "/home/u/llama.cpp/build-sm75")
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=75", sh)
        self.assertIn("grep -v '^GGML_NATIVE'", sh)
        self.assertNotIn("rm ", sh)

    def test_emb_latency_reports_error_without_server(self):
        inv = self.inv()
        inv["servers"]["emb"]["port"] = 1   # nada escucha ahí
        r = modelos.emb_latency(3, inv, load=False)
        self.assertIn("error", r)

    # ---------------------------------------------------------------- config y consola
    def test_apply_choice_sets_process_config(self):
        C.apply_server_choice("stm", {"model": QWEN, "gpu": "0", "ctx": 65536})
        self.assertEqual((C.REMOTE_MODEL_PATH_STM, C.GPU1_INDEX, C.SLOT_CTX_GPU1), (QWEN, "0", 32768))
        C.apply_server_choice("emb", {"gpu": "2"})
        self.assertEqual(C.GPU2_INDEX, "2")
        C.apply_server_choice("ltm", {"model": QWEN})
        self.assertIn(f"-m {QWEN}", servidor._launch_cmd("0", 8087, 32768, 1, "/tmp/p", "/tmp/l",
                                                          C.REMOTE_MODEL_PATH_LTM))

    def test_console_has_panel_and_endpoint(self):
        html = (ROOT / "interfaz" / "index.html").read_text(encoding="utf-8")
        self.assertIn("openServers()", html)
        self.assertIn("/api/servers/switch", html)
        fe = (ROOT / "interfaz" / "frontend.py").read_text(encoding="utf-8")
        self.assertIn('"/api/servers/switch"', fe)


if __name__ == "__main__":
    unittest.main()
