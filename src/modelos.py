# -*- coding: utf-8 -*-
"""[PARCHE31] Elegir qué modelo carga cada servidor y en qué GPU.

Lo usan el panel «🧩 Modelos y GPUs» de la consola y scripts/modelos.py. Por SSH (o en local si SERVER_IP es
127.0.0.1) hace tres cosas:

  inventory()  lee, sin tocar nada, las GPU (nvidia-smi), qué proceso escucha en cada puerto (8087 LTM, 8088 STM,
               8089 embeddings), con qué línea de órdenes y en qué GPU corre, y los .gguf de las carpetas de modelos.
  plan()       valida una elección (modelo de la lista, GPU que existe, contexto razonable, cabe en la VRAM) y
               construye la orden: LA MISMA línea con la que corre ahora el servidor, cambiando solo -m, -c y la
               GPU, para no perder ningún ajuste tuyo (--alias, -np, caché KV, hilos...).
  switch()     para el servidor de ese puerto, arranca el nuevo y espera a /health. Si el nuevo muere o no
               responde, vuelve a arrancar el anterior con su orden exacta. Guarda la elección en
               BODY/servidores.json (pisa al .env; ver config.py).

El alias (--alias) no cambia: el agente sigue hablando con «el modelo» de cada puerto sin reconfigurar nada.
El modelo de embeddings no se cambia desde aquí (invalidaría los vectores ya guardados): solo su GPU.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

import config as C
import servidor as S

ROLES = ("stm", "ltm", "emb")
LABELS = {"stm": "Agente · responde (STM)", "ltm": "Memoria larga · ingesta, sueño y juez (LTM)",
          "emb": "Embeddings · búsqueda semántica"}
EMB_HINT = re.compile(r"(bge|embed|e5-|gte-|nomic|rerank|minilm|mxbai|jina)", re.I)
SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
M_ARGS, C_ARGS, NP_ARGS = ("-m", "--model"), ("-c", "--ctx-size"), ("-np", "--parallel")
_CACHE: dict[str, str] = {}


class NeedsForce(ValueError):
    """La elección parece no caber en la VRAM: se puede forzar."""


# ------------------------------------------------------------------ configuración por papel
def role_cfg(role: str) -> dict[str, Any]:
    if role == "ltm":
        return {"port": int(C.SERVER_PORT_GPU0), "gpu": str(C.GPU0_INDEX), "ctx": int(C.REMOTE_CTX_GPU0), "slots": 1,
                "model": getattr(C, "REMOTE_MODEL_PATH_LTM", "") or C.REMOTE_MODEL_PATH,
                "pid_file": C.REMOTE_PID_GPU0, "log": C.REMOTE_LOG_GPU0, "flag": "_started_gpu0"}
    if role == "stm":
        slots = max(1, int(C.PARALLEL_SLOTS_GPU1))
        return {"port": int(C.SERVER_PORT_GPU1), "gpu": str(C.GPU1_INDEX), "ctx": int(C.SLOT_CTX_GPU1) * slots,
                "slots": slots, "model": getattr(C, "REMOTE_MODEL_PATH_STM", "") or C.REMOTE_MODEL_PATH,
                "pid_file": C.REMOTE_PID_GPU1, "log": C.REMOTE_LOG_GPU1, "flag": "_started_gpu1"}
    if role == "emb":
        return {"port": int(C.SERVER_PORT_GPU2), "gpu": str(C.GPU2_INDEX), "ctx": int(C.EMB_CTX), "slots": 1,
                "model": C.EMB_MODEL_PATH, "pid_file": C.REMOTE_PID_GPU2, "log": C.REMOTE_LOG_GPU2,
                "flag": "_started_gpu2"}
    raise ValueError(f"papel desconocido: {role!r} (usa stm, ltm o emb)")


def models_dirs(extra: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Carpetas donde buscar .gguf: REMOTE_MODELS_DIR o la carpeta madre de los modelos conocidos."""
    raw = str(getattr(C, "REMOTE_MODELS_DIR", "") or "")
    dirs = [d.strip() for d in raw.split(",") if d.strip()]
    if not dirs:
        for m in [role_cfg(r)["model"] for r in ROLES] + list(extra):
            if m and "/" in str(m):
                parent = str(Path(m).parent.parent)
                if parent not in ("/", ".", "") and parent not in dirs:
                    dirs.append(parent)
    return dirs


# ------------------------------------------------------------------ argumentos de llama-server
def get_arg(argv: list[str], names: tuple[str, ...]) -> str | None:
    for i, a in enumerate(argv):
        if a in names and i + 1 < len(argv):
            return argv[i + 1]
        for n in names:
            if n.startswith("--") and a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def set_arg(argv: list[str], names: tuple[str, ...], value: Any) -> list[str]:
    out = list(argv)
    for i, a in enumerate(out):
        if a in names and i + 1 < len(out):
            out[i + 1] = str(value)
            return out
        for n in names:
            if n.startswith("--") and a.startswith(n + "="):
                out[i] = f"{n}={value}"
                return out
    return out + [names[0], str(value)]


def _int_or(v: Any, default: int | None = None) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ inventario (solo lectura)
_CACHE_GREP = "grep -E '^(CMAKE_CUDA_ARCHITECTURES|GGML_NATIVE)[:=]'"


def configured_bins() -> list[str]:
    """Binarios que conviene inspeccionar: el del .env y los propios de cada GPU."""
    out = []
    for b in [C.LLAMA_SERVER_BIN, *getattr(C, "GPU_SERVER_BINS", {}).values()]:
        if b and b not in out:
            out.append(b)
    return out


def _script(ports: list[int], dirs: list[str], with_state: bool = True, bins: list[str] | None = None) -> str:
    q = shlex.quote
    lines: list[str] = []
    if with_state:
        q6 = "index,pci.bus_id,name,memory.used,memory.total,compute_cap"
        lines += [
            "echo '##GPUS'",
            # [PARCHE32] compute_cap (drivers antiguos no lo tienen: entonces sin él)
            f"if out=$(nvidia-smi --query-gpu={q6} --format=csv,noheader,nounits 2>/dev/null); then "
            "echo \"$out\" | sed 's/^/c6, /'; else "
            "nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.total --format=csv,noheader,nounits "
            "2>/dev/null | sed 's/^/c5, /'; fi",
            "echo '##APPS'",
            "nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits 2>/dev/null",
        ]
        for p in ports:
            # [l]/[-]: el patrón no se encuentra a sí mismo en la línea de órdenes de este mismo shell
            lines += [
                f"echo '##PORT {p}'",
                f"pid=$(ss -ltnpH 'sport = :{p}' 2>/dev/null | grep -o 'pid=[0-9]*' | head -n1 | cut -d= -f2)",
                f"[ -z \"$pid\" ] && pid=$(lsof -nP -tiTCP:{p} -sTCP:LISTEN 2>/dev/null | head -n1)",
                f"[ -z \"$pid\" ] && pid=$(pgrep -f -- '[-]-port {p}( |$)' 2>/dev/null | head -n1)",
                "echo \"pid=$pid\"",
                "if [ -n \"$pid\" ] && [ -r /proc/$pid/cmdline ]; then "
                "echo \"exe=$(readlink /proc/$pid/exe 2>/dev/null)\"; "
                "tr '\\0' '\\n' < /proc/$pid/cmdline | sed 's/^/arg=/'; "
                "tr '\\0' '\\n' < /proc/$pid/environ 2>/dev/null | grep -E '^CUDA_(VISIBLE_DEVICES|DEVICE_ORDER)=' "
                "| sed 's/^/env=/'; "
                # [PARCHE32] arquitecturas CUDA con las que se compiló ese llama-server (build/CMakeCache.txt)
                f"{_CACHE_GREP} \"$(dirname \"$(dirname \"$(readlink /proc/$pid/exe)\")\")/CMakeCache.txt\" 2>/dev/null "
                "| sed 's/^/cache=/'; fi",
            ]
        for b in bins or []:
            lines += [f"echo {q('##BIN ' + b)}",
                      f"p=$(command -v {q(b)} 2>/dev/null || echo {q(b)}); echo \"path=$p\"; "
                      f"{_CACHE_GREP} \"$(dirname \"$(dirname \"$p\")\")/CMakeCache.txt\" 2>/dev/null | sed 's/^/cache=/'"]
    lines.append("echo '##MODELS'")
    for d in dirs:
        lines.append(f"find {q(d)} -maxdepth 4 -type f -name '*.gguf' 2>/dev/null | while IFS= read -r f; do "
                     "printf '%s\\t%s\\n' \"$(wc -c < \"$f\" | tr -d ' ')\" \"$f\"; done")
    lines.append("echo '##END'")
    return "\n".join(lines)


def _bus(b: str) -> str:
    b = b.strip().lower()
    return b.split(":", 1)[1] if b.count(":") == 2 else b


def _parse_cache(lines: list[str]) -> dict[str, Any]:
    """{'archs': ['86', '89'] | None, 'native': bool | None} a partir de las líneas de CMakeCache.txt."""
    info: dict[str, Any] = {"archs": None, "native": None}
    for ln in lines:
        k, _, v = ln.partition("=")
        name = k.split(":", 1)[0]
        if name == "CMAKE_CUDA_ARCHITECTURES":
            info["archs"] = [a.strip() for a in v.split(";") if a.strip()]
        elif name == "GGML_NATIVE":
            info["native"] = v.strip().upper() in ("ON", "1", "TRUE", "YES")
    return info


def arch_ok(cap: str | None, info: dict[str, Any] | None) -> bool | None:
    """¿Corre un llama-server compilado para info['archs'] en una GPU con esa compute capability? None = no se sabe."""
    if not cap or not info or not info.get("archs"):
        return None
    try:
        cc = int(round(float(cap) * 10))
    except ValueError:
        return None
    for a in info["archs"]:
        m = re.fullmatch(r"(\d+)(?:-(real|virtual))?", a.strip().lower())
        if not m:
            return None   # native / all / all-major: depende de la máquina donde se compiló
        n, kind = int(m.group(1)), m.group(2)
        if (cc // 10 == n // 10 and cc >= n) if kind == "real" else cc >= n:
            return True
    return False


def parse_inventory(text: str) -> dict[str, Any]:
    inv: dict[str, Any] = {"gpus": [], "apps": [], "ports": {}, "files": [], "bins": {}}
    sec, cur, binfo = None, None, None
    for raw in str(text).splitlines():
        line = raw.rstrip("\r")
        if line.startswith("##"):
            tag = line[2:].strip()
            if tag.startswith("PORT "):
                sec, cur = "port", {"pid": None, "exe": "", "argv": [], "env": {}, "cache": []}
                inv["ports"][int(tag[5:])] = cur
            elif tag.startswith("BIN "):
                sec, binfo = "bin", {"name": tag[4:].strip(), "path": "", "cache": []}
                inv["bins"][binfo["name"]] = binfo
            else:
                sec = tag.lower()
            continue
        if sec == "gpus":
            parts = [x.strip() for x in line.split(",")]
            fmt = parts.pop(0) if parts and parts[0] in ("c5", "c6") else "c5"
            cap = parts.pop() if fmt == "c6" and len(parts) >= 6 else None
            if len(parts) >= 5 and parts[0].isdigit():
                inv["gpus"].append({"index": parts[0], "bus": _bus(parts[1]), "name": ", ".join(parts[2:-2]),
                                    "used_mb": _int_or(parts[-2], 0), "total_mb": _int_or(parts[-1], 0),
                                    "cap": cap if cap and re.fullmatch(r"\d+\.\d+", cap) else None})
        elif sec == "apps":
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                inv["apps"].append({"pid": parts[0], "bus": _bus(parts[1]), "used_mb": _int_or(parts[2], 0)})
        elif sec == "port" and cur is not None:
            if line.startswith("pid="):
                cur["pid"] = line[4:].strip() or None
            elif line.startswith("exe="):
                cur["exe"] = line[4:].strip()
            elif line.startswith("arg="):
                cur["argv"].append(line[4:])
            elif line.startswith("env=") and "=" in line[4:]:
                k, v = line[4:].split("=", 1)
                cur["env"][k] = v
            elif line.startswith("cache="):
                cur["cache"].append(line[6:])
        elif sec == "bin" and binfo is not None:
            if line.startswith("path="):
                binfo["path"] = line[5:].strip()
            elif line.startswith("cache="):
                binfo["cache"].append(line[6:])
        elif sec == "models" and "\t" in line:
            size, path = line.split("\t", 1)
            inv["files"].append((_int_or(size, 0) or 0, path.strip()))
    if inv["ports"]:
        for p in inv["ports"].values():
            while p["argv"] and p["argv"][-1] == "":
                p["argv"].pop()
    builds: dict[str, dict[str, Any]] = {}
    for p in inv["ports"].values():
        if p["exe"]:
            builds[p["exe"]] = dict(_parse_cache(p["cache"]), path=p["exe"])
    for name, b in inv["bins"].items():
        info = dict(_parse_cache(b["cache"]), path=b["path"] or name)
        builds.setdefault(name, info)
        if b["path"]:
            builds.setdefault(b["path"], info)
    inv["builds"] = builds
    return inv


def binary_for(srv: dict[str, Any], gpu: str | None) -> str:
    """Qué llama-server usar para ese servidor en esa GPU: el propio de la GPU (GPU_SERVER_BINS), si no el que ya
    usa (salvo que sea el propio de otra GPU), y si no el del .env."""
    bins = getattr(C, "GPU_SERVER_BINS", {}) or {}
    if gpu is not None and str(gpu) in bins:
        return bins[str(gpu)]
    cur = srv.get("exe") or (srv.get("argv") or [""])[0]
    if cur and cur not in bins.values():
        return cur
    return C.LLAMA_SERVER_BIN


def build_hint(gpu: dict[str, Any]) -> str:
    arch = (gpu.get("cap") or "?").replace(".", "")
    return (f"Compila un llama-server para ella (python3 scripts/modelos.py --compilar-gpu {gpu['index']}: sm_{arch}, "
            f"en una carpeta aparte, sin tocar el tuyo) y añade al .env  GPU_SERVER_BINS={gpu['index']}=<ruta que te dé>.")


def list_models(files: list[tuple[int, str]], dirs: list[str]) -> list[dict[str, Any]]:
    """.gguf de chat y de embeddings; los troceados se muestran una vez (su primera parte) con el tamaño total."""
    sizes: dict[str, int] = {}
    parts: dict[str, int] = {}
    for size, path in files:
        m = SHARD.search(path)
        key = SHARD.sub("", path) if m else path
        sizes[key] = sizes.get(key, 0) + size
        parts[key] = parts.get(key, 0) + 1
    out, seen = [], set()
    for size, path in sorted(files, key=lambda x: x[1].lower()):
        name = Path(path).name
        if "mmproj" in name.lower():
            continue
        m = SHARD.search(path)
        if m and m.group(1) != "00001":
            continue
        key = SHARD.sub("", path) if m else path
        if key in seen:
            continue
        seen.add(key)
        root = next((d for d in dirs if path.startswith(d.rstrip("/") + "/")), "")
        rel = path[len(root.rstrip("/")) + 1:] if root else name
        out.append({"path": path, "name": rel, "file": name, "mb": round(sizes[key] / 1048576),
                    "gb": round(sizes[key] / 1e9, 1), "parts": parts[key],
                    "kind": "emb" if EMB_HINT.search(path) else "llm"})
    return out


def _ssh_hint(stderr: str) -> str:
    s = stderr.lower()
    if "permission denied" in s or "publickey" in s:
        return (f"Entra por SSH con contraseña, pero la consola necesita entrar SIN contraseña. En el Mac: "
                f"ssh-copy-id {C.SERVER_USER}@{C.SERVER_IP}")
    if "timed out" in s or "no route" in s or "refused" in s or "could not resolve" in s:
        return f"El servidor {C.SERVER_IP} no responde por SSH: ¿está encendido y en la red?"
    return "Revisa SERVER_IP y SERVER_USER en tu .env."


def inventory(run: Callable[..., Any] | None = None, health: Callable[[int], bool] | None = None) -> dict[str, Any]:
    run = run or S.run_remote
    health = health or S.is_remote_health_ok
    cfgs = {r: role_cfg(r) for r in ROLES}
    dirs = models_dirs()
    base: dict[str, Any] = {"host": C.SERVER_IP, "local": S._is_local(), "gpus": [], "servers": {}, "models": [],
                            "dirs": dirs, "warnings": [], "choices": C.load_server_choices()}
    try:
        res = run(f"bash -c {shlex.quote(_script([c['port'] for c in cfgs.values()], dirs, bins=configured_bins()))}",
                  timeout=40)
    except Exception as exc:
        return dict(base, error=f"No puedo ejecutar órdenes en {C.SERVER_IP}: {type(exc).__name__}: {exc}",
                    hint=_ssh_hint(str(exc)))
    out = getattr(res, "stdout", "") or ""
    if "##END" not in out:
        err = (getattr(res, "stderr", "") or "").strip()
        return dict(base, error=f"No puedo ejecutar órdenes en {C.SERVER_IP} (código {getattr(res, 'returncode', '?')}): "
                    f"{err[-300:] or 'sin salida'}", hint=_ssh_hint(err))
    inv = parse_inventory(out)
    files = list(inv["files"])
    running = [get_arg(p["argv"], M_ARGS) or "" for p in inv["ports"].values()]
    more = [d for d in models_dirs(running) if d not in dirs]
    if more:   # carpetas que solo se conocen por los servidores en marcha
        try:
            r2 = run(f"bash -c {shlex.quote(_script([], more, with_state=False))}", timeout=40)
            files += parse_inventory(getattr(r2, "stdout", "") or "")["files"]
            dirs = dirs + more
        except Exception:
            pass
    gpus = inv["gpus"]
    by_bus = {g["bus"]: g for g in gpus}
    for g in gpus:
        g["servers"] = []
    apps = {a["pid"]: a for a in inv["apps"]}
    builds = inv.get("builds", {})
    for role, cfg in cfgs.items():
        p = inv["ports"].get(cfg["port"], {"pid": None, "exe": "", "argv": [], "env": {}, "cache": []})
        argv, pid = p["argv"], p["pid"]
        app = apps.get(str(pid)) if pid else None
        g = by_bus.get(app["bus"]) if app else None
        model = get_arg(argv, M_ARGS) or (cfg["model"] if not pid else "")
        srv = {"role": role, "label": LABELS[role], "port": cfg["port"], "up": bool(health(cfg["port"])),
               "pid": pid, "exe": p["exe"], "argv": argv, "env": p["env"], "argv_ok": bool(argv),
               "model": model, "model_name": Path(model).name if model else "?",
               "gpu": g["index"] if g else None, "gpu_name": g["name"] if g else "",
               "vram_mb": app["used_mb"] if app else None,
               "ctx": _int_or(get_arg(argv, C_ARGS), None), "slots": _int_or(get_arg(argv, NP_ARGS), None),
               "config": {"model": cfg["model"], "gpu": cfg["gpu"], "ctx": cfg["ctx"]},
               "env_default": dict(C.ENV_SERVER_DEFAULTS.get(role, {})),
               "chosen": role in base["choices"]}
        # [PARCHE32] ¿en qué GPU puede arrancar este servidor con el llama-server que le tocaría?
        srv["gpu_ok"] = {x["index"]: arch_ok(x.get("cap"), builds.get(binary_for(srv, x["index"]))) for x in gpus}
        srv["archs"] = (builds.get(p["exe"]) or {}).get("archs")
        if g:
            g["servers"].append(role.upper())
        if pid and not argv:
            base["warnings"].append(f"{role.upper()}: veo el proceso {pid} en el puerto {cfg['port']}, pero no su línea "
                                    f"de órdenes (¿de otro usuario o no es Linux?): al cambiarlo se arrancará con la "
                                    f"configuración del .env.")
        if pid and argv and not g and gpus:
            base["warnings"].append(f"{role.upper()}: no sé en qué GPU está el proceso {pid} (nvidia-smi no lo lista).")
        base["servers"][role] = srv
    if not gpus:
        base["warnings"].append("nvidia-smi no responde: no puedo mostrar las GPU ni comprobar la VRAM.")
    for x in gpus:
        bad = [r.upper() for r, s in base["servers"].items() if s["gpu_ok"].get(x["index"]) is False]
        x["incompatible"] = bad
        if bad:
            arch = next((s["archs"] for s in base["servers"].values() if s.get("archs")), None)
            base["warnings"].append(
                f"GPU {x['index']} ({x['name']}, compute {x.get('cap')}): tu llama-server está compilado para "
                f"{';'.join(arch or ['?'])} y no arranca en ella. " + build_hint(x))
    base.update(gpus=gpus, models=list_models(files, dirs), dirs=dirs, builds=builds)
    if not base["models"]:
        base["warnings"].append("No encuentro ningún .gguf en " + (", ".join(dirs) or "ninguna carpeta") +
                                ": define REMOTE_MODELS_DIR en tu .env.")
    return base


# ------------------------------------------------------------------ plan (validación + orden)
def _launch(env_prefix: str, argv: list[str], log_file: str, pid_file: str, append: bool = False) -> str:
    q = shlex.quote
    red = ">>" if append else ">"
    return (f"{env_prefix} nohup {' '.join(q(a) for a in argv)} {red} {q(log_file)} 2>&1 < /dev/null &\n"
            f"echo $! > {q(pid_file)}; echo \"pid=$!\"")


def _env_prefix(env: dict[str, str] | None, gpu: str | None) -> str:
    q = shlex.quote
    if gpu is not None:
        return f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={q(str(gpu))}"
    # tal cual estaba (env -u: sin heredar variables CUDA de este shell)
    keep = " ".join(f"{k}={q(v)}" for k, v in sorted((env or {}).items()))
    return ("env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_ORDER " + keep).rstrip()


def resolve_model(query: str, models: list[dict[str, Any]], kind: str = "llm") -> str:
    """Ruta exacta, o un trozo único del nombre («qwen3.8»)."""
    q = str(query or "").strip()
    if not q:
        raise ValueError("falta el modelo")
    for m in models:
        if m["path"] == q:
            return q
    ql = q.lower()
    pool = [m for m in models if m["kind"] == kind]
    exact = [m for m in pool if ql in (m["name"].lower(), m["file"].lower(), Path(m["file"]).stem.lower())]
    if len(exact) == 1:
        return exact[0]["path"]
    hits = [m for m in pool if ql in m["name"].lower()]
    if len(hits) == 1:
        return hits[0]["path"]
    if not hits:
        raise ValueError(f"no encuentro ningún modelo que contenga «{q}»")
    # [PARCHE32] «GSQ-RCO-IQ3_S» → el archivo que TERMINA así, no el «…IQ3_S-mtp»
    ends = [m for m in hits if Path(m["file"]).stem.lower().endswith(ql)]
    if len(ends) == 1:
        return ends[0]["path"]
    raise ValueError(f"«{q}» coincide con {len(hits)} modelos: " + ", ".join(m["name"] for m in hits[:6]) +
                     " (usa un trozo que solo esté en uno, o la ruta completa)")


def plan(role: str, inv: dict[str, Any], model: str | None = None, gpu: str | int | None = None,
         ctx: int | str | None = None, force: bool = False) -> dict[str, Any]:
    cfg = role_cfg(role)
    srv = inv["servers"][role]
    models = inv.get("models", [])
    gpus = inv.get("gpus", [])
    # modelo
    if role == "emb":
        if model and model != srv.get("model"):
            raise ValueError("El modelo de embeddings no se cambia desde aquí: los vectores guardados dejarían de "
                             "servir. Solo puedes moverlo de GPU.")
        new_model = srv.get("model") or cfg["model"]
    else:
        new_model = str(model or srv.get("model") or cfg["model"] or "")
        known = {m["path"]: m for m in models}
        if new_model not in known and new_model != srv.get("model"):
            raise ValueError(f"«{new_model}» no está entre los .gguf del servidor ({', '.join(inv.get('dirs', []))}).")
        if known.get(new_model, {}).get("kind") == "emb":
            raise ValueError(f"«{Path(new_model).name}» parece un modelo de embeddings, no de chat.")
    if not new_model:
        raise ValueError("No sé qué modelo cargar: elige uno o define la ruta en tu .env.")
    # GPU
    new_gpu: str | None
    if gpu is None or str(gpu).strip() == "":
        new_gpu = srv.get("gpu") if srv.get("gpu") is not None else (None if srv.get("argv_ok") else cfg["gpu"])
    else:
        new_gpu = str(gpu).strip()
        if not new_gpu.isdigit():
            raise ValueError(f"GPU no válida: {gpu!r}")
        if gpus and new_gpu not in {g["index"] for g in gpus}:
            raise ValueError(f"No hay GPU {new_gpu} (hay: {', '.join(g['index'] for g in gpus)}).")
    # contexto
    old_ctx = srv.get("ctx") or cfg["ctx"]
    if role == "emb" or ctx in (None, ""):
        new_ctx = int(old_ctx)
    else:
        new_ctx = _int_or(ctx, -1) or -1
        if not 2048 <= new_ctx <= 1048576:
            raise ValueError(f"Contexto no válido: {ctx!r} (entre 2048 y 1048576 tokens)")
    # [PARCHE32] binario para esa GPU y si sus arquitecturas CUDA la cubren
    new_bin = binary_for(srv, new_gpu)
    gx = next((x for x in gpus if x["index"] == new_gpu), None) if new_gpu is not None else None
    if gx is not None and arch_ok(gx.get("cap"), (inv.get("builds") or {}).get(new_bin)) is False:
        archs = ";".join(((inv.get("builds") or {}).get(new_bin) or {}).get("archs") or ["?"])
        raise ValueError(f"La GPU {new_gpu} ({gx['name']}, compute {gx.get('cap')}) no puede usar {new_bin}: está "
                         f"compilado solo para {archs}. " + build_hint(gx))
    # ¿cambia algo?
    same = (srv.get("pid") and srv.get("up") and new_model == srv.get("model") and new_ctx == int(old_ctx)
            and (new_gpu is None or new_gpu == srv.get("gpu")))
    if same:
        raise ValueError("Ya está así: mismo modelo, misma GPU y mismo contexto.")
    # VRAM
    need_mb = free_mb = None
    g = None
    if gpus and new_gpu is not None:
        g = next((x for x in gpus if x["index"] == new_gpu), None)
    if g is not None:
        free_mb = int(g["total_mb"]) - int(g["used_mb"])
        if srv.get("gpu") == new_gpu and srv.get("vram_mb"):
            free_mb += int(srv["vram_mb"])
        size_mb = next((m["mb"] for m in models if m["path"] == new_model), None)
        if role == "emb" and srv.get("vram_mb"):
            need_mb = int(srv["vram_mb"])
        elif size_mb:
            need_mb = int(size_mb) + (512 if role == "emb" else 1024)
        if need_mb and need_mb > free_mb and not force:
            raise NeedsForce(f"No parece caber: {Path(new_model).name} necesita unos {need_mb / 1024:.1f} GB (sin contar "
                             f"la caché del contexto) y en la GPU {new_gpu} ({g['name']}) quedarían {free_mb / 1024:.1f} GB libres.")
    # orden: la línea actual con -m/-c/GPU cambiados; si no se conoce, la del .env
    if srv.get("argv_ok"):
        argv = [new_bin] + list(srv["argv"][1:])
        if role != "emb":
            argv = set_arg(argv, M_ARGS, new_model)
            if get_arg(argv, C_ARGS) is not None or new_ctx != int(old_ctx):
                argv = set_arg(argv, C_ARGS, new_ctx)
        env_new = _env_prefix(srv.get("env"), new_gpu)
        rollback = _launch(_env_prefix(srv.get("env"), None), [srv.get("exe") or srv["argv"][0]] + list(srv["argv"][1:]),
                           cfg["log"], cfg["pid_file"], append=True)
    else:
        if role == "emb":
            argv = S.emb_argv()
        else:
            argv = S.llm_argv(cfg["port"], new_ctx, cfg["slots"], new_model)
        argv = [new_bin] + argv[1:]
        env_new = _env_prefix(None, new_gpu if new_gpu is not None else cfg["gpu"])
        rollback = None   # no hay orden anterior conocida
    if srv.get("pid") and not srv.get("argv_ok"):
        rollback = None
    gpu_txt = f"GPU {new_gpu}" if new_gpu is not None else "la GPU de ahora"
    if new_bin != (srv.get("exe") or ""):
        gpu_txt += f" · binario {new_bin}"
    return {"role": role, "model": new_model, "gpu": new_gpu, "ctx": new_ctx, "argv": argv,
            "cmd": _launch(env_new, argv, cfg["log"], cfg["pid_file"]), "rollback": rollback,
            "need_mb": need_mb, "free_mb": free_mb,
            "summary": f"{role.upper()} → {Path(new_model).name} · {gpu_txt} · contexto {new_ctx}"}


# ------------------------------------------------------------------ cambio
def props(port: int) -> dict[str, Any]:
    try:
        url = f"http://{S._health_host()}:{port}/props"
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return {}


def refresh_cache() -> dict[str, str]:
    for role in ("stm", "ltm"):
        mp = props(role_cfg(role)["port"]).get("model_path") or ""
        if mp:
            _CACHE[f"{role}_model"] = Path(str(mp)).name
    return dict(_CACHE)


def cached() -> dict[str, str]:
    return dict(_CACHE)


def save_choice(role: str, model: str | None, gpu: str | None, ctx: int | None) -> dict[str, Any]:
    """Guarda solo lo que difiere del .env; si todo coincide, borra la entrada."""
    data = C.load_server_choices()
    env = C.ENV_SERVER_DEFAULTS.get(role, {})
    entry: dict[str, Any] = {}
    if role != "emb" and model and model != env.get("model"):
        entry["model"] = model
    if gpu is not None and str(gpu) != str(env.get("gpu")):
        entry["gpu"] = str(gpu)
    if role != "emb" and ctx and int(ctx) != int(env.get("ctx") or 0):
        entry["ctx"] = int(ctx)
    if entry:
        data[role] = entry
    else:
        data.pop(role, None)
    path = Path(C.SERVERS_CHOICE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return data


def _ev(kind: str, text: str, **kw: Any) -> dict[str, Any]:
    d = {"event": kind, ("error" if kind == "error" else "text"): text}
    d.update(kw)
    return d


def _pid_of(stdout: str) -> str | None:
    m = re.search(r"pid=(\d+)", stdout or "")
    return m.group(1) if m else None


def _stop(run: Callable[..., Any], pid: str) -> tuple[bool, str]:
    sh = (f"pid={int(pid)}; kill -0 $pid 2>/dev/null || {{ echo GONE; exit 0; }}; "
          "kill $pid 2>/dev/null || { echo NOPERM; exit 3; }; "
          "for i in $(seq 1 40); do kill -0 $pid 2>/dev/null || break; sleep 0.5; done; "
          "if kill -0 $pid 2>/dev/null; then kill -9 $pid 2>/dev/null; sleep 1; fi; "
          "kill -0 $pid 2>/dev/null && { echo ALIVE; exit 4; }; sleep 1; echo STOPPED")
    res = run(f"bash -c {shlex.quote(sh)}", timeout=40)
    out = (getattr(res, "stdout", "") or "").strip()
    if "NOPERM" in out:
        return False, f"no tengo permiso para parar el proceso {pid} (¿es de otro usuario?)"
    if "ALIVE" in out:
        return False, f"el proceso {pid} no se deja parar"
    if "STOPPED" in out or "GONE" in out:
        return True, ""
    return False, f"no sé si el proceso {pid} paró: {out or (getattr(res, 'stderr', '') or '').strip()[-200:]}"


def _wait(port: int, pid: str | None, log_file: str, run: Callable[..., Any], health: Callable[[int], bool],
          sleep: Callable[[float], None], timeout_s: float) -> Iterator[dict[str, Any]]:
    """Eventos de progreso; el último es {'event': '_wait', 'ok': bool, 'why': str}."""
    t0 = time.time()
    last_note = t0
    n = 0
    while time.time() - t0 < timeout_s:
        if health(port):
            yield {"event": "_wait", "ok": True, "why": ""}
            return
        n += 1
        if pid and n % 3 == 0:
            r = run(f"kill -0 {int(pid)} 2>/dev/null && echo VIVO || echo MUERTO", timeout=10)
            if "MUERTO" in (getattr(r, "stdout", "") or ""):
                tail = run(f"tail -n 25 {shlex.quote(log_file)}", timeout=10)
                yield {"event": "_wait", "ok": False,
                       "why": "el proceso terminó al cargar.\n--- log ---\n" + (getattr(tail, "stdout", "") or "").strip()[-1500:]}
                return
        if time.time() - last_note >= 10:
            last_note = time.time()
            yield _ev("step", f"cargando pesos… {int(time.time() - t0)} s")
        sleep(2)
    tail = run(f"tail -n 25 {shlex.quote(log_file)}", timeout=10)
    yield {"event": "_wait", "ok": False, "why": f"no respondió en {int(timeout_s)} s.\n--- log ---\n" +
           (getattr(tail, "stdout", "") or "").strip()[-1500:]}


def switch(role: str, model: str | None = None, gpu: str | int | None = None, ctx: int | str | None = None,
           force: bool = False, save: bool = True, run: Callable[..., Any] | None = None,
           health: Callable[[int], bool] | None = None, sleep: Callable[[float], None] = time.sleep,
           timeout_s: float | None = None) -> Iterator[dict[str, Any]]:
    """Generador de eventos {'event': 'step'|'done'|'error', ...}. Nunca deja el puerto sin servidor si puede evitarlo."""
    run = run or S.run_remote
    health = health or S.is_remote_health_ok
    timeout_s = float(timeout_s or max(180.0, float(getattr(C, "SERVER_HEALTH_TIMEOUT_S", 180.0))))
    try:
        cfg = role_cfg(role)
    except ValueError as exc:
        yield _ev("error", str(exc))
        return
    yield _ev("step", f"leyendo el estado de {C.SERVER_IP}…")
    inv = inventory(run, health)
    if inv.get("error"):
        yield _ev("error", inv["error"], hint=inv.get("hint", ""))
        return
    try:
        p = plan(role, inv, model, gpu, ctx, force)
    except NeedsForce as exc:
        yield _ev("error", str(exc), need_force=True)
        return
    except ValueError as exc:
        yield _ev("error", str(exc))
        return
    old = inv["servers"][role]
    yield _ev("step", "plan: " + p["summary"])
    if old.get("pid"):
        yield _ev("step", f"parando el servidor actual (proceso {old['pid']}: {old.get('model_name', '?')})…")
        ok, why = _stop(run, str(old["pid"]))
        if not ok:
            yield _ev("error", f"No cambio nada: {why}.")
            return
    yield _ev("step", "arrancando: " + Path(p["model"]).name + (f" en la GPU {p['gpu']}" if p["gpu"] is not None else ""))
    res = run(p["cmd"], timeout=20)
    new_pid = _pid_of(getattr(res, "stdout", ""))
    result: dict[str, Any] = {}
    if new_pid is None:
        result = {"ok": False, "why": "no se pudo lanzar: " + ((getattr(res, "stderr", "") or "").strip()[-300:] or "sin pid")}
    else:
        for ev in _wait(cfg["port"], new_pid, cfg["log"], run, health, sleep, timeout_s):
            if ev["event"] == "_wait":
                result = ev
            else:
                yield ev
    if result.get("ok"):
        loaded = str(props(cfg["port"]).get("model_path") or "")
        if role != "emb" and loaded and Path(loaded).name != Path(p["model"]).name:
            yield _ev("step", f"⚠ el servidor dice que ha cargado {Path(loaded).name}")
        if save:
            C.apply_server_choice(role, {"model": p["model"] if role != "emb" else None,
                                         "gpu": p["gpu"], "ctx": p["ctx"] if role != "emb" else None})
            save_choice(role, p["model"], p["gpu"], p["ctx"])
        # quien paraba el anterior al cerrar la consola, para también este (y viceversa)
        setattr(S, cfg["flag"], bool(getattr(S, cfg["flag"], False)) if old.get("pid") else True)
        refresh_cache()
        yield _ev("done", f"{p['summary']} · listo (proceso {new_pid or '?'})" +
                  (" · elección guardada" if save else ""), plan={k: p[k] for k in ("role", "model", "gpu", "ctx")},
                  previous={"model": old.get("model"), "gpu": old.get("gpu"), "ctx": old.get("ctx")})
        return
    yield _ev("step", "⚠ el nuevo servidor no arrancó: " + result.get("why", "?"))
    if new_pid:
        _stop(run, new_pid)
    if not p.get("rollback"):
        msg = ("El nuevo modelo no arrancó (antes no había servidor en ese puerto). " if not old.get("pid") else
               "El nuevo modelo no arrancó y no conozco la orden del anterior para volver a él: arráncalo desde aquí "
               "con el modelo anterior. ")
        yield _ev("error", msg + result.get("why", ""), rolled_back=False)
        return
    yield _ev("step", "vuelvo al servidor anterior con su orden exacta…")
    res = run(p["rollback"], timeout=20)
    back_pid = _pid_of(getattr(res, "stdout", ""))
    back: dict[str, Any] = {}
    for ev in _wait(cfg["port"], back_pid, cfg["log"], run, health, sleep, timeout_s):
        if ev["event"] == "_wait":
            back = ev
        else:
            yield ev
    refresh_cache()
    if back.get("ok"):
        yield _ev("error", f"No se pudo cargar {Path(p['model']).name}; el servidor anterior "
                  f"({old.get('model_name', '?')}) vuelve a estar activo. Motivo: " +
                  result.get("why", "?").split("\n", 1)[0], rolled_back=True, log=result.get("why", ""))
    else:
        yield _ev("error", "No se pudo cargar el nuevo modelo NI volver al anterior: revisa el servidor. " +
                  back.get("why", ""), rolled_back=False)


def switch_guarded(agent: Any, role: str, **kw: Any) -> Iterator[dict[str, Any]]:
    """switch() con el agente en pausa: sin chat, sin ingesta, sin ciclo de sueño (y sin compactar si es el STM)."""
    if agent is None:
        yield from switch(role, **kw)
        return
    chat_lock = getattr(agent, "_chat_lock", None)
    ltm_lock = getattr(agent, "ltm_lock", None)
    stm = getattr(agent, "stm", None)
    compact_lock = getattr(stm, "_compact_lock", None) if role == "stm" else None
    if chat_lock is not None and not chat_lock.acquire(timeout=10):
        yield _ev("error", "El agente está respondiendo o ingiriendo: inténtalo cuando termine.")
        return
    paused = hasattr(agent, "_pause_ltm")
    try:
        if paused:
            agent._pause_ltm()
        with contextlib.ExitStack() as stack:
            for lock, what in ((ltm_lock, "el ciclo de sueño"), (compact_lock, "la compactación del corto plazo")):
                if lock is None:
                    continue
                if not lock.acquire(timeout=120):
                    yield _ev("error", f"{what} no termina su paso: inténtalo en un minuto.")
                    return
                stack.callback(lock.release)
            log = getattr(agent, "log", None)
            if log:
                log(f"🧩 Cambio de servidor {role.upper()} pedido desde la consola…")
            for ev in switch(role, **kw):
                if log and ev.get("event") in ("done", "error"):
                    log(("✓ " + ev.get("text", "")) if ev["event"] == "done" else ("⚠️ " + ev.get("error", "")))
                yield ev
            if hasattr(agent, "server_ready"):
                try:
                    agent.server_ready = all(S.is_remote_health_ok(role_cfg(r)["port"]) for r in ("stm", "ltm"))
                except Exception:
                    pass
    finally:
        if paused:
            agent._resume_ltm()
        if chat_lock is not None:
            chat_lock.release()


# ------------------------------------------------------------------ [PARCHE32] latencia de los embeddings
def _post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _stats(ms: list[float]) -> dict[str, Any] | None:
    if not ms:
        return None
    s = sorted(ms)
    return {"n": len(s), "p50": round(s[len(s) // 2], 1), "p95": round(s[min(len(s) - 1, int(len(s) * 0.95))], 1),
            "media": round(sum(s) / len(s), 1)}


def emb_latency(n: int = 40, inv: dict[str, Any] | None = None, load: bool = True) -> dict[str, Any]:
    """Milisegundos por llamada de embeddings (1 texto corto, como una consulta del agente) donde estén ahora,
    en reposo y mientras el modelo de chat que comparte su GPU está generando (lo que pasa si el agente o la
    compactación usan ese servidor a la vez)."""
    import threading
    inv = inv or inventory()
    if inv.get("error"):
        return {"error": inv["error"]}
    emb = inv["servers"]["emb"]
    gpu = emb.get("gpu")
    neighbour = next((r for r in ("stm", "ltm") if gpu is not None and inv["servers"][r].get("gpu") == gpu
                      and inv["servers"][r].get("up")), None)
    host = S._health_host()
    url = f"http://{host}:{emb['port']}/v1/embeddings"
    texts = [f"Dato {i}: en {1890 + i % 60} el vapor número {i} atracó en el muelle {i % 7} de Maliaño con "
             f"{i * 3} toneladas de carga" for i in range(n + 5)]

    def call(text: str) -> float:
        t0 = time.perf_counter()
        _post(url, {"model": C.EMB_MODEL_ALIAS, "input": [text]}, 30)
        return (time.perf_counter() - t0) * 1000

    try:
        for t in texts[:3]:
            call(t)
        idle = [call(t) for t in texts[3:3 + n]]
    except Exception as exc:
        return {"error": f"embeddings en :{emb['port']} no responden: {type(exc).__name__}: {exc}"}
    loaded: list[float] = []
    gen_err = ""
    if load and neighbour:
        port = inv["servers"][neighbour]["port"]
        stop = threading.Event()
        started = threading.Event()

        def generate() -> None:
            nonlocal gen_err
            while not stop.is_set():
                try:
                    started.set()
                    _post(f"http://{host}:{port}/v1/chat/completions",
                          {"model": C.REMOTE_MODEL_ALIAS, "max_tokens": 300, "temperature": 0.8, "cache_prompt": False,
                           "chat_template_kwargs": {"enable_thinking": False},
                           "messages": [{"role": "user", "content": "Escribe una crónica larga y detallada del puerto "
                                                                    "de Santander en el siglo XIX."}]}, 180)
                except Exception as exc:  # pragma: no cover - red
                    gen_err = f"{type(exc).__name__}: {exc}"
                    return

        th = threading.Thread(target=generate, daemon=True)
        th.start()
        started.wait(5)
        time.sleep(1.5)   # que la generación esté en marcha
        try:
            loaded = [call(t) for t in texts[3:3 + n]]
        finally:
            stop.set()
            th.join(240)
    return {"gpu": gpu, "gpu_name": emb.get("gpu_name", ""), "neighbour": neighbour, "idle": _stats(idle),
            "load": _stats(loaded), "gen_error": gen_err}


# ------------------------------------------------------------------ [PARCHE32] compilar llama-server para una GPU
def build_script(src_build: str, arch: str, jobs: int = 8) -> tuple[str, str, str]:
    """(orden, carpeta nueva, log). Copia las opciones GGML_/LLAMA_ y los compiladores de TU compilación y cambia
    solo la arquitectura CUDA; compila en <fuente>/build-sm<arch>, sin tocar build/."""
    q = shlex.quote
    src = str(Path(src_build).parent)
    out = f"{src}/build-sm{arch}"
    logf = f"/tmp/hmem-build-sm{arch}.log"
    sh = (
        f"set -e; B={q(src_build)}; SRC={q(src)}; O={q(out)}; L={q(logf)}; "
        "test -f \"$B/CMakeCache.txt\" || { echo \"NOCACHE $B\"; exit 3; }; "
        "CM=$(grep -E '^CMAKE_COMMAND:INTERNAL=' \"$B/CMakeCache.txt\" | cut -d= -f2-); CM=${CM:-cmake}; "
        "OPTS=$(grep -E '^(GGML_[A-Z0-9_]+:BOOL|LLAMA_[A-Z0-9_]+:BOOL|CMAKE_CUDA_COMPILER:FILEPATH|"
        "CMAKE_CUDA_HOST_COMPILER:FILEPATH|CMAKE_C_COMPILER:FILEPATH|CMAKE_CXX_COMPILER:FILEPATH)=' "
        "\"$B/CMakeCache.txt\" | grep -v '^GGML_NATIVE' | sed 's/^/-D/' | tr '\\n' ' '); "
        "J=$(nproc 2>/dev/null || echo 4); "
        f"[ \"$J\" -gt {int(jobs)} ] && J={int(jobs)}; "
        "echo \"cmake $OPTS\" > \"$L\"; "
        f"\"$CM\" -S \"$SRC\" -B \"$O\" $OPTS -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES={q(arch)} "
        "-DCMAKE_BUILD_TYPE=Release >> \"$L\" 2>&1 || { echo FAILCONF; tail -n 25 \"$L\"; exit 4; }; "
        "\"$CM\" --build \"$O\" --config Release -j \"$J\" --target llama-server >> \"$L\" 2>&1 "
        "|| { echo FAILBUILD; tail -n 25 \"$L\"; exit 5; }; "
        "test -x \"$O/bin/llama-server\" && echo \"BUILT=$O/bin/llama-server\""
    )
    return sh, out, logf


def compile_for_gpu(gpu: str, inv: dict[str, Any] | None = None, run: Callable[..., Any] | None = None,
                    timeout: int = 5400) -> dict[str, Any]:
    run = run or S.run_remote
    inv = inv or inventory(run)
    if inv.get("error"):
        return {"ok": False, "error": inv["error"]}
    g = next((x for x in inv["gpus"] if x["index"] == str(gpu)), None)
    if g is None or not g.get("cap"):
        return {"ok": False, "error": f"no conozco la GPU {gpu} o su compute capability"}
    exe = next((s["exe"] for s in inv["servers"].values() if s.get("exe") and "/bin/" in s["exe"]), "") or C.LLAMA_SERVER_BIN
    if "/bin/" not in exe:
        return {"ok": False, "error": f"no sé dónde está la compilación de {exe}"}
    arch = g["cap"].replace(".", "")
    sh, out, logf = build_script(str(Path(exe).parent.parent), arch)
    t0 = time.time()
    res = run(f"bash -c {shlex.quote(sh)}", timeout=timeout)
    text = (getattr(res, "stdout", "") or "") + (getattr(res, "stderr", "") or "")
    m = re.search(r"BUILT=(\S+)", text)
    if m:
        return {"ok": True, "path": m.group(1), "arch": arch, "min": round((time.time() - t0) / 60, 1), "log": logf}
    return {"ok": False, "arch": arch, "log": logf, "error": text.strip()[-1500:] or "sin salida"}
