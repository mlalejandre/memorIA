# Arranque/parada de llama-server (local o por SSH).
from __future__ import annotations

import os
import shlex
import subprocess
import time
import urllib.request

import config as C

_started_gpu0 = False
_started_gpu1 = False
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", ""}

_STOP_SH = (
    'if [ -f {pf} ]; then pid=$(cat {pf}); '
    'if kill -0 "$pid" 2>/dev/null; then kill "$pid"; sleep 2; '
    'kill -0 "$pid" 2>/dev/null && kill -9 "$pid"; fi; rm -f {pf}; fi'
)


def _is_local() -> bool:
    return C.SERVER_IP.strip().lower() in _LOCAL_HOSTS


def _health_host() -> str:
    return "127.0.0.1" if _is_local() else C.SERVER_IP


def run_remote(command: str, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    """Ejecuta `command` en local (SERVER_IP=127.0.0.1) o por SSH."""
    if _is_local():
        argv = ["bash", "-c", command]
    else:
        argv = ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                f"{C.SERVER_USER}@{C.SERVER_IP}", command]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


run_ssh = run_remote  # alias de compatibilidad


def is_remote_health_ok(port: int) -> bool:
    try:
        url = f"http://{_health_host()}:{port}/health"
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=2.0) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False


def _wait_for_server(port: int, gpu_idx: str, log_file: str) -> None:
    deadline = time.time() + C.SERVER_HEALTH_TIMEOUT_S
    print(f"⏳ Esperando a llama-server (GPU {gpu_idx}, puerto {port})...", flush=True)
    last = 0.0
    while time.time() < deadline:
        if is_remote_health_ok(port):
            print(f"✅ llama-server (GPU {gpu_idx}) listo.", flush=True)
            return
        now = time.time()
        if now - last >= 10.0:
            print(f"   Cargando pesos en GPU {gpu_idx}... ({int(deadline - now)}s restantes)", flush=True)
            last = now
        time.sleep(2)
    peek = ""
    try:
        peek = run_remote(f"tail -n 30 {shlex.quote(log_file)}").stdout.strip()
    except Exception:
        pass
    raise RuntimeError(f"Servidor GPU {gpu_idx} no respondió.\n--- Log ---\n{peek}")


def llm_argv(port: int, ctx: int, slots: int, model_path: str | None = None) -> list[str]:
    """Argumentos de llama-server para un servidor de chat (también los usa modelos.py)."""
    host = "127.0.0.1" if _is_local() else "0.0.0.0"
    return [C.LLAMA_SERVER_BIN, "-m", model_path or C.REMOTE_MODEL_PATH, "--alias", C.REMOTE_MODEL_ALIAS,
            "--host", host, "--port", str(port), "-ngl", "all", "--split-mode", "none", "--flash-attn", "on",
            "-c", str(ctx), "-np", str(slots), "-ctk", "q8_0", "-ctv", "q8_0", "--threads", str(C.REMOTE_THREADS),
            *shlex.split(C.LLAMA_SERVER_EXTRA)]


def _launch_cmd(gpu_idx: str, port: int, ctx: int, slots: int, pid_file: str, log_file: str,
                model_path: str | None = None) -> str:
    q = shlex.quote
    # CUDA_DEVICE_ORDER=PCI_BUS_ID: los índices de GPU coinciden con los de nvidia-smi
    return (
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={q(gpu_idx)} nohup "
        f"{' '.join(q(a) for a in llm_argv(port, ctx, slots, model_path))} "
        f"> {q(log_file)} 2>&1 < /dev/null &\n"
        f"echo $! > {q(pid_file)}"
    )


def _start(label: str, gpu_idx: str, port: int, ctx: int, slots: int, pid: str, log: str,
           model_path: str | None = None) -> bool:
    """Devuelve True si ESTA sesión lanzó el servidor."""
    if is_remote_health_ok(port):
        print(f"✅ {label} ya activa en puerto {port}.", flush=True)
        return False
    if not C.SERVER_AUTOSTART:
        raise RuntimeError(
            f"{label}: no hay ningún servidor LLM en el puerto {port} y SERVER_AUTOSTART=0. "
            f"Arráncalo tú (llama-server, vLLM, Ollama...) o define SERVER_AUTOSTART=1 en .env.")
    if _is_local() and os.name == "nt":
        raise RuntimeError(
            "El autoarranque local de llama-server requiere un entorno POSIX (Linux, macOS o WSL2). "
            "En Windows usa SERVER_AUTOSTART=0 y arranca el servidor tú, o apunta SERVER_IP a un host Linux (SSH).")
    if not (model_path or C.REMOTE_MODEL_PATH):
        raise RuntimeError("SERVER_AUTOSTART=1 requiere REMOTE_MODEL_PATH (ruta al .gguf) en .env.")
    print(f"⚡ Lanzando llama-server para {label}...", flush=True)
    res = run_remote(_launch_cmd(gpu_idx, port, ctx, slots, pid, log, model_path), timeout=10)
    if res.returncode != 0:
        raise RuntimeError(f"Error lanzando {label}: {res.stderr.strip()}")
    _wait_for_server(port, gpu_idx, log)
    return True


def start_gpu0() -> None:
    global _started_gpu0
    if _start("GPU 0 (Memoria Larga)", C.GPU0_INDEX, C.SERVER_PORT_GPU0, C.REMOTE_CTX_GPU0, 1,
              C.REMOTE_PID_GPU0, C.REMOTE_LOG_GPU0,
              getattr(C, "REMOTE_MODEL_PATH_LTM", "") or None):
        _started_gpu0 = True


def start_gpu1() -> None:
    global _started_gpu1
    total_ctx = C.SLOT_CTX_GPU1 * C.PARALLEL_SLOTS_GPU1
    if _start("GPU 1 (Memoria Corta / Agente)", C.GPU1_INDEX, C.SERVER_PORT_GPU1, total_ctx,
              C.PARALLEL_SLOTS_GPU1, C.REMOTE_PID_GPU1, C.REMOTE_LOG_GPU1,
              getattr(C, "REMOTE_MODEL_PATH_STM", "") or None):
        _started_gpu1 = True


# GPU 2: servidor de embeddings (opcional: si falla, el sistema sigue sin él)
_started_gpu2 = False


def _port_open(host: str, port: int) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def emb_argv() -> list[str]:
    """Argumentos de llama-server para el servidor de embeddings."""
    host = "127.0.0.1" if _is_local() else "0.0.0.0"
    ctx = str(C.EMB_CTX)
    return [C.LLAMA_SERVER_BIN, "-m", C.EMB_MODEL_PATH, "--alias", C.EMB_MODEL_ALIAS, "--host", host,
            "--port", str(C.SERVER_PORT_GPU2), "-ngl", "all", "--split-mode", "none", "--embeddings",
            "--pooling", "cls", "-c", ctx, "-b", ctx, "-ub", ctx, "-np", "1", "--threads", str(C.REMOTE_THREADS),
            *shlex.split(getattr(C, "EMB_SERVER_EXTRA", ""))]


def _launch_emb_cmd() -> str:
    q = shlex.quote
    return (
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={q(C.GPU2_INDEX)} nohup "
        f"{' '.join(q(a) for a in emb_argv())} "
        f"> {q(C.REMOTE_LOG_GPU2)} 2>&1 < /dev/null &\n"
        f"echo $! > {q(C.REMOTE_PID_GPU2)}"
    )


def start_gpu2() -> None:
    global _started_gpu2
    if not getattr(C, "EMB_ENABLED", True):
        return
    port = C.SERVER_PORT_GPU2
    try:
        if is_remote_health_ok(port) or _port_open(_health_host(), port):
            print(f"✅ Servidor de embeddings (GPU 2) ya activo en puerto {port}.", flush=True)
            return
        if not C.SERVER_AUTOSTART:
            print(f"ℹ️ Sin servidor de embeddings en el puerto {port}: búsqueda semántica desactivada "
                  f"hasta que lo arranques (o SERVER_AUTOSTART=1 + EMB_MODEL_PATH).", flush=True)
            return
        if _is_local() and os.name == "nt":
            print("ℹ️ El autoarranque local de embeddings requiere POSIX; arráncalo tú.", flush=True)
            return
        if not C.EMB_MODEL_PATH:
            print("ℹ️ Define EMB_MODEL_PATH (.gguf de bge-m3) en .env para autoarrancar los embeddings.", flush=True)
            return
        print("⚡ Lanzando llama-server de embeddings (GPU 2)...", flush=True)
        res = run_remote(_launch_emb_cmd(), timeout=10)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip())
        _wait_for_server(port, C.GPU2_INDEX, C.REMOTE_LOG_GPU2)
        _started_gpu2 = True
    except Exception as exc:
        print(f"⚠️ Embeddings no disponibles ({exc}); el sistema seguirá sin ellos.", flush=True)


def start_all_servers() -> None:
    start_gpu0()
    start_gpu1()
    start_gpu2()


def stop_all_servers() -> None:
    """Para SOLO los servidores que lanzó esta sesión (TERM y luego KILL si hace falta)."""
    if C.SERVER_KEEP_ALIVE:
        print("\nℹ️ SERVER_KEEP_ALIVE=1: los servidores siguen activos.", flush=True)
        return
    pids = [p for started, p in ((_started_gpu0, C.REMOTE_PID_GPU0), (_started_gpu1, C.REMOTE_PID_GPU1),
                                 (_started_gpu2, getattr(C, "REMOTE_PID_GPU2", ""))) if started and p]
    if not pids:
        return
    print("\n🛑 Liberando VRAM de los servidores lanzados por esta sesión...", flush=True)
    script = "; ".join(_STOP_SH.format(pf=shlex.quote(p)) for p in pids)
    try:
        run_remote(script, timeout=15)
        print("✓ VRAM liberada.", flush=True)
    except Exception as exc:
        print(f"Aviso al liberar VRAM: {exc}", flush=True)


# Alias de compatibilidad
start_remote_server_if_needed = start_all_servers
stop_remote_server = stop_all_servers
