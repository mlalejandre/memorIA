#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[PARCHE31] Modelos y GPUs de los servidores desde el terminal del Mac (lo mismo que el panel de la consola).

    python3 scripts/modelos.py                      # qué hay: GPU, servidores, .gguf disponibles (no toca nada)
    python3 scripts/modelos.py stm GSQ-RCO-IQ3_S    # el agente responde con Qwen3.8 IQ3_S (trozo único del nombre o ruta)
    python3 scripts/modelos.py stm GSQ-RCO-IQ3_S --ctx 65536 --gpu 1
    python3 scripts/modelos.py emb --gpu 2          # embeddings a la GPU 2
    python3 scripts/modelos.py ltm nail             # memoria larga con nail-35b
    python3 scripts/modelos.py stm GSQ-RCO-IQ3_S --no-guardar   # prueba sin guardar la elección

Con la consola ABIERTA usa mejor su panel «🧩 Modelos y GPUs»: pone en pausa el agente mientras cambia.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
import modelos  # noqa: E402


def show(inv: dict) -> int:
    if inv.get("error"):
        print("❌ " + inv["error"])
        if inv.get("hint"):
            print("💡 " + inv["hint"])
        return 2
    print(f"Servidor {inv['host']}")
    for g in inv["gpus"]:
        print(f"  GPU {g['index']} · {g['name']:<28} {g['used_mb'] / 1024:5.1f} / {g['total_mb'] / 1024:4.1f} GB · "
              f"{', '.join(g['servers']) or '—'}")
    for role in modelos.ROLES:
        s = inv["servers"][role]
        state = "activo" if s["up"] else ("no responde" if s["pid"] else "parado")
        print(f"  {role.upper():<3} :{s['port']} {state:<11} {s['model_name']} · GPU {s['gpu'] if s['gpu'] is not None else '?'}"
              f" · ctx {s['ctx'] or '?'} · slots {s['slots'] or '?'}" + (" · elegido en la consola" if s["chosen"] else ""))
    print(f"Modelos ({', '.join(inv['dirs'])}):")
    for m in inv["models"]:
        print(f"  {'emb' if m['kind'] == 'emb' else '   '} {m['gb']:6.1f} GB  {m['name']}" + (f"  ({m['parts']} partes)" if m["parts"] > 1 else ""))
    for w in inv["warnings"]:
        print("⚠ " + w)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Modelos y GPUs de los servidores")
    ap.add_argument("role", nargs="?", choices=modelos.ROLES)
    ap.add_argument("model", nargs="?", help="ruta o trozo único del nombre del .gguf")
    ap.add_argument("--gpu")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--forzar", action="store_true", help="aunque parezca no caber en la VRAM")
    ap.add_argument("--no-guardar", action="store_true", help="no guardar la elección (BODY/servidores.json)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    inv = modelos.inventory()
    if not a.role:
        if a.json:
            print(json.dumps(inv, ensure_ascii=False, indent=2))
            return 0
        return show(inv)
    if inv.get("error"):
        return show(inv)
    model = None
    if a.model:
        try:
            model = modelos.resolve_model(a.model, inv["models"], "emb" if a.role == "emb" else "llm")
        except ValueError as exc:
            print("❌ " + str(exc))
            return 2
    code = 1
    for ev in modelos.switch(a.role, model=model, gpu=a.gpu, ctx=a.ctx, force=a.forzar, save=not a.no_guardar):
        if ev["event"] == "step":
            print("· " + ev["text"], flush=True)
        elif ev["event"] == "done":
            print("✓ " + ev["text"])
            code = 0
        else:
            print("❌ " + ev["error"] + (" (añade --forzar para intentarlo igualmente)" if ev.get("need_force") else ""))
    if code == 0 and not a.no_guardar:
        print(f"  (guardado en {C.SERVERS_CHOICE_FILE}; la consola lo usará al arrancar)")
    return code


if __name__ == "__main__":
    sys.exit(main())
