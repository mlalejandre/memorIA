#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLANTILLA DE PARCHE · memoria-unificada (motor usado en los parches 19-33).

Cómo se usa (lo hace Claude, no el usuario):
  1. Reconstruye el proyecto del volcado (scripts/estructura.py) en un repo git y haz el cambio a mano.
  2. git add … && python3 scripts/generar_edits.py <repo> edits.json   → EDITS mínimos, verificados byte a byte.
  3. Copia esta plantilla como parcheNN.py: cambia PATCH_ID, REQUIRES y applied(); sustituye __EDITS__ y
     __NEW_FILES__ por el contenido de edits.json (repr de Python); escribe las FASES (al final) y la cabecera
     con QUÉ CAMBIA y la REGLA DE DECISIÓN fijada ANTES de medir.
  4. Pruébalo en una copia limpia del estado del usuario: tests + emulado + rama --live contra
     scripts/servidor_falso.py + --revertir. Entrégalo; el usuario lo guarda como parche.py y lo ejecuta con el venv.

El motor: cada EDIT exige que el bloque viejo exista EXACTAMENTE una vez (todo o nada); copia los archivos en
.parche_backup/<PATCH_ID>/ con manifest; py_compile y reversión automática si algo falla; log en
resultados_parches/; --revertir deshace. Termina siempre con «📋 RESUMEN PARA CLAUDE».
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PATCH_ID = "parcheNN"
MARK = "parcheNN"
ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / ".parche_backup" / PATCH_ID
LOGDIR = ROOT / "resultados_parches"
REQUIRES = [
    # (archivo, texto que DEBE existir antes de aplicar, qué es)
    ("src/modelos.py", "def compile_for_gpu", "v0.3.0a5"),
]
EDITS = __EDITS__
NEW_FILES = __NEW_FILES__

_LOG = None


def log(msg: str = "") -> None:
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "-"


# ==============================================================================
# MOTOR: coincidencia EXACTA de un bloque (líneas comparadas tras rstrip), todo o nada
# ==============================================================================
def _find(lines: list[str], block: list[str]) -> list[int]:
    r = [x.rstrip() for x in block]
    rc = [x.rstrip() for x in lines]
    n = len(r)
    return [i for i in range(len(rc) - n + 1) if rc[i:i + n] == r]


def _apply_one(text: str, old: str, new: str) -> tuple[str | None, int]:
    lines = text.split("\n")
    hits = _find(lines, old.split("\n"))
    if len(hits) != 1:
        return None, len(hits)
    i = hits[0]
    return "\n".join(lines[:i] + new.split("\n") + lines[i + len(old.split("\n")):]), 1


def applied() -> bool:
    """Cámbialo por algo que este parche AÑADE de verdad (una función, una variable)."""
    return False


def apply_patch() -> bool:
    for f, marker, what in REQUIRES:
        p = ROOT / f
        if not p.exists() or marker not in p.read_text(encoding="utf-8"):
            log(f"❌ Falta {marker} en {f} ({what}): este parche va sobre otra versión del proyecto.")
            return False
    texts: dict[str, str] = {}
    for f, old, new, desc in EDITS:
        if f not in texts:
            texts[f] = (ROOT / f).read_text(encoding="utf-8")
        res, n = _apply_one(texts[f], old, new)
        if res is None:
            log(f"❌ {desc}: el bloque original aparece {n} veces (se esperaba 1). No se ha tocado nada.")
            log("   ¿Trabajas sobre el volcado de v0.3.0a3 sin cambios locales? (git status)")
            return False
        texts[f] = res
    for f in NEW_FILES:
        if (ROOT / f).exists():
            log(f"❌ {f} ya existe y no es de este parche. No se ha tocado nada.")
            return False
    # copia de seguridad + manifest
    if BACKUP.exists():
        shutil.rmtree(BACKUP)
    BACKUP.mkdir(parents=True)
    manifest = {"patch": PATCH_ID, "fecha": _dt.datetime.now().isoformat(timespec="seconds"),
                "modificados": {}, "nuevos": sorted(NEW_FILES)}
    for f in texts:
        dst = BACKUP / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst)
        manifest["modificados"][f] = {"antes": sha(ROOT / f)}
    for f, text in texts.items():
        (ROOT / f).write_text(text, encoding="utf-8")
        manifest["modificados"][f]["despues"] = sha(ROOT / f)
    for f, text in NEW_FILES.items():
        (ROOT / f).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / f).write_text(text, encoding="utf-8")
    (BACKUP / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = []
    for f in list(texts) + list(NEW_FILES):
        if f.endswith(".py"):
            try:
                py_compile.compile(str(ROOT / f), doraise=True)
            except py_compile.PyCompileError as exc:
                bad.append(f"{f}: {exc.msg}")
    if bad:
        log("❌ Error de compilación; revierto:\n   " + "\n   ".join(bad))
        revert()
        return False
    log(f"✓ Parche aplicado: {len(EDITS)} cambios en {len(texts)} archivos + {len(NEW_FILES)} nuevo(s). "
        f"Copia en {BACKUP.relative_to(ROOT)}")
    return True


def revert() -> bool:
    mf = BACKUP / "manifest.json"
    if not mf.exists():
        log("❌ No hay copia de seguridad de este parche.")
        return False
    m = json.loads(mf.read_text(encoding="utf-8"))
    for f in m["modificados"]:
        shutil.copy2(BACKUP / f, ROOT / f)
    for f in m["nuevos"]:
        (ROOT / f).unlink(missing_ok=True)
    log(f"↩ Revertido {PATCH_ID}: {len(m['modificados'])} archivos restaurados, {len(m['nuevos'])} eliminado(s).")
    return True


# ==============================================================================
# EJECUCIÓN DE PRUEBAS Y BENCHMARK
# ==============================================================================
def run(cmd: list[str], env: dict[str, str] | None = None, timeout: int | None = None) -> tuple[int, str]:
    e = dict(os.environ)
    e.update(env or {})
    e["PYTHONUNBUFFERED"] = "1"
    log("$ " + " ".join(f"{k}={v}" for k, v in (env or {}).items()) + (" " if env else "") + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    out = []
    t0 = time.time()
    for line in p.stdout:
        out.append(line)
        log(line.rstrip("\n"))
        if timeout and time.time() - t0 > timeout:
            p.kill()
            break
    return p.wait(), "".join(out)


def effective_config(extra: dict[str, str]) -> dict[str, str]:
    code = ("import sys; sys.path.insert(0, 'src'); import config as C, json; print(json.dumps({k: str(getattr(C, k)) "
            "for k in ('RETRIEVAL_PROFILE','RERANK_MODE','LTM_DIRECT_TOP_K','EPI_PROMPT_K','EPI_LOOKUP_K',"
            "'LTM_DECOMPOSE','EPI_DIVERSITY','LTM_SECOND_PASS','ANSWER_DIRECT','AGENT_THINK','REMOTE_MODEL_ALIAS','EMB_MODEL_ALIAS','SERVER_IP')}))")
    e = dict(os.environ)
    e.update(extra)
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=e, capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": p.stderr.strip()[-300:]}


def cache_root() -> Path:
    return Path(os.getenv("HMEM_LOCOMO_DIR") or (ROOT / "experiment" / "locomo_cache"))


def latest(pattern: str, after: float = 0.0) -> Path | None:
    files = [p for p in cache_root().glob(pattern) if p.stat().st_mtime >= after]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def pct(x: float) -> str:
    return f"{x * 100:.1f} %"


# ==============================================================================
# FASES (adáptalas: tests y emulado son obligatorios; --live según el experimento)
# ==============================================================================
OFF = {"LTM_DECOMPOSE": "0", "EPI_DIVERSITY": "off", "LTM_SECOND_PASS": "0", "ANSWER_DIRECT": "0",
       "HMEM_ANSWER_WORKERS": "1", "HMEM_ANSWER_CACHE_PROMPT": "0"}


def phase_tests() -> bool:
    rc, out = run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    m = re.search(r"Ran (\d+) tests", out)
    ok = rc == 0 and bool(out.strip()) and "OK" in out.splitlines()[-1]
    log(f"{'✓' if ok else '❌'} tests: {m.group(1) if m else '?'} ejecutados · {'OK' if ok else 'FALLAN'}")
    return ok


def phase_emulado() -> bool:
    rc, out = run([sys.executable, "benchmarks/prueba.py", "--only", "locomo,seguridad"], dict(OFF))
    ok = rc == 0 and "Lo recuperado llega al prompt real del agente" in out
    log(f"{'✓' if ok else '❌'} emulado: {'OK' if ok else 'FALLA'}")
    return ok


def resumen(t_start: float, cfg: dict, tests_ok, emu_ok, decision: str) -> None:
    log("\n" + "=" * 96)
    log("📋 RESUMEN PARA CLAUDE · " + PATCH_ID + " · " + _dt.datetime.now().strftime("%d/%m/%Y %H:%M") +
        f" · {(time.time() - t_start) / 60:.0f} min")
    log("=" * 96)
    log("Config EFECTIVA (entorno + .env): " + " · ".join(f"{k}={v}" for k, v in cfg.items()))
    log(f"Tests: {'OK' if tests_ok else 'no ejecutados' if tests_ok is None else 'FALLAN'} · "
        f"Emulado: {'OK' if emu_ok else 'no ejecutado' if emu_ok is None else 'FALLA'}")
    log("REGLA: (cópiala de la cabecera)")
    log(f"⇒ DECISIÓN: {decision}")
    log("=" * 96)


def main() -> int:
    global _LOG
    ap = argparse.ArgumentParser(description=PATCH_ID)
    ap.add_argument("--solo-eval", action="store_true")
    ap.add_argument("--revertir", action="store_true")
    a = ap.parse_args()
    if not (ROOT / "pyproject.toml").exists() or 'name = "hierarchical-memory"' not in (ROOT / "pyproject.toml").read_text():
        print("❌ Ejecuta el parche desde la raíz de memoria-unificada.")
        return 2
    LOGDIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    _LOG = (LOGDIR / f"{PATCH_ID}_{stamp}.log").open("w", encoding="utf-8")
    t_start = time.time()
    if a.revertir:
        return 0 if revert() else 1
    tests_ok = emu_ok = None
    if not a.solo_eval:
        if not applied():
            if not apply_patch():
                return 1
            tests_ok = phase_tests()
            if not tests_ok:
                log("❌ Los tests fallan con el parche: revierto.")
                revert()
                return 1
        emu_ok = phase_emulado()
    decision = "SIN MEDIR"
    # … fases --live del experimento, pareadas contra la base reproducible …
    resumen(t_start, effective_config(dict(OFF)), tests_ok, emu_ok, decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
