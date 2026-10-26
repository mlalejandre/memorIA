from __future__ import annotations

import json
import tarfile
import time
from datetime import datetime, timezone
from typing import Any, Callable

from config import (
    BACKUP_DIR,
    LLM_BASE_URL,
    LLM_MODEL,
    LOG_DIR,
    MEM_BACKUP_EVERY,
    MEM_BACKUP_KEEP,
    MEM_IDLE_MAX_SLEEP_S,
    MEM_IDLE_SLEEP_S,
)
from cartuchos import CartridgeManager
from largo_plazo import MemLLM, MemoryTree

HELP = (
    "Escribe un recuerdo, una instrucción o una pregunta.\n"
    "  /arbol    muestra el árbol de memoria\n"
    "  /stats    estadísticas\n"
    "  /ayuda    esta ayuda\n"
    "  (línea vacía) o /seguir    reanuda el bucle\n"
    "  /salir o Ctrl+C otra vez    termina el programa"
)

def _log(record: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "memoria.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def _backup(tick: int, tree: MemoryTree | None = None) -> None:
    """Copia consistente de la BD con la API de backup de SQLite (válida con WAL)."""
    import gzip
    import shutil
    import sqlite3

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUP_DIR / f"memoria_tick_{tick:07d}.db.gz"
    tmp = BACKUP_DIR / f".memoria_tick_{tick:07d}.db.tmp"
    try:
        if tree is None:
            raise RuntimeError("sin árbol de memoria")
        dest = sqlite3.connect(str(tmp))
        try:
            with tree.lock:
                tree.conn.backup(dest)
        finally:
            dest.close()
        with tmp.open("rb") as src, gzip.open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        tmp.unlink(missing_ok=True)
        old = sorted(BACKUP_DIR.glob("memoria_tick_*.db.gz"))[:-max(1, MEM_BACKUP_KEEP)]
        for f in old:
            f.unlink(missing_ok=True)
        print(f"📦 [BACKUP] {target.name}", flush=True)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"⚠️ [BACKUP] fallo: {exc}", flush=True)

def chat(tree: MemoryTree) -> bool:
    """Modo conversación interactiva."""
    print("\n⏸  BUCLE PAUSADO. " + HELP, flush=True)
    while True:
        try:
            line = input("\nTú> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        low = line.lower()
        if not line or low in ("/seguir", "/s"):
            print("▶️  Reanudando bucle...", flush=True)
            return True
        if low in ("/salir", "/q", "/exit"):
            return False
        if low in ("/ayuda", "/help"):
            print(HELP)
            continue
        if low == "/arbol":
            print(tree.render())
            continue
        if low == "/stats":
            print(tree.summary())
            continue
        try:
            print(tree.process_user_message(line), flush=True)
        except KeyboardInterrupt:
            print("\n(cancelado)", flush=True)
        except Exception as exc:
            print(f"⚠️ Error: {type(exc).__name__}: {exc}", flush=True)

def run_loop(start_server: Callable[[], None], stop_server: Callable[[], None]) -> None:
    try:
        start_server()
        cartridge_path = CartridgeManager().get_primary_path()
        tree = MemoryTree(cartridge_path, MemLLM())
        print("=" * 72)
        print("AGENTE DE MEMORIA ITERATIVO INFINITO")
        print(f"Memoria:  {cartridge_path}")
        print(f"Backend:  {LLM_BASE_URL} ({LLM_MODEL})")
        print(f"Estado:   {tree.summary()}")
        print("Ctrl+C  => pausa el bucle para darle un recuerdo, instrucción o pregunta.")
        print("=" * 72, flush=True)

        idle = 0
        add = 1
        last_bucket = tree.tick // MEM_BACKUP_EVERY if MEM_BACKUP_EVERY else 0
        last_backup_rev = tree.rev
        base_sleep = max(0.1, MEM_IDLE_SLEEP_S)
        max_sleep = max(base_sleep, MEM_IDLE_MAX_SLEEP_S)

        while True:
            try:
                tick = tree.advance(add)
                started = time.perf_counter()
                name, report, worked = tree.maintenance_step()
                elapsed = time.perf_counter() - started
                failed = report.startswith("error")

                if worked or failed:
                    idle = 0 if worked else idle + 1
                    print(f"🔧 [tick {tick:06d}] {name:<10} {report}  ({elapsed:.1f}s)", flush=True)
                    _log({
                        "timestamp": datetime.now(timezone.utc).isoformat(), "tick": tick,
                        "task": name, "report": report, "elapsed": round(elapsed, 2),
                    })
                else:
                    idle += 1
                    if idle == 8:
                        print(f"💤 [tick {tick:06d}] Memoria estable: reposo con pausas crecientes "
                              f"(hasta {max_sleep:.0f}s). Ctrl+C para hablar.", flush=True)

                if MEM_BACKUP_EVERY and tick // MEM_BACKUP_EVERY > last_bucket:
                    last_bucket = tick // MEM_BACKUP_EVERY
                    if tree.rev != last_backup_rev:
                        _backup(tick, tree)
                        last_backup_rev = tree.rev

                delay = min(max_sleep, base_sleep * 2 ** min(idle // 8, 6))
                add = max(1, round(delay / base_sleep))
                time.sleep(delay)
            except KeyboardInterrupt:
                if not chat(tree):
                    print("\nDetenido por el usuario.", flush=True)
                    break
                idle, add = 0, 1

        if tree.rev != last_backup_rev:
            _backup(tree.tick, tree)
    finally:
        stop_server()
