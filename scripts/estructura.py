#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Volcado del proyecto en un solo archivo de texto (proyecto_completo.txt) para retomar el trabajo con Claude.

    python3 scripts/estructura.py                       # todo el código y la documentación
    python3 scripts/estructura.py --ligero              # sin los archivos más grandes (banco, tests, consola, datos)
    python3 scripts/estructura.py --solo src/agente.py,src/config.py   # solo esos archivos (para un parche concreto)

Qué entra: SOLO lo que git conoce (archivos versionados + nuevos no ignorados), es decir, lo mismo que verías en
GitHub. Así nunca se cuelan body/ (tus memorias), .env, cachés, resultados ni copias de parches, aunque
el .gitignore cambie. Sin git, recorre la carpeta con las mismas exclusiones.
La cabecera dice versión, commit y si hay cambios sin commit: el Claude que lo lea sabe exactamente sobre qué
estado está trabajando (los parches exigen coincidencia exacta del código).
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "proyecto_completo.txt"

# Nunca, aunque git los conociera.
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "logs", "snapshots",
                 "backups", ".parche_backup", "resultados_parches", "locomo_cache", "body", "node_modules"}
EXCLUDED_FILES = {OUTPUT_FILE.name, ".DS_Store", ".env", "sesion.json", "_ltm_pendiente.jsonl", "_archivo.jsonl",
                  "servidores.json"}
EXCLUDED_SUFFIXES = (".bak", ".pre_parche", ".tmp", ".db", ".sqlite", ".gz", ".tar", ".pyc")
TEXT_EXTENSIONS = {".py", ".txt", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
                   ".sh", ".bash", ".zsh", ".xml", ".html", ".css", ".js", ".ts", ".example"}
ALWAYS_INCLUDE = {"Dockerfile", "Makefile", "requirements.txt", ".gitignore", "LICENSE"}
# --ligero: lo más voluminoso y menos necesario para conversar (se puede pedir luego con --solo).
HEAVY = {"benchmarks/prueba.py", "benchmarks/locomo.py", "interfaz/index.html", "tests/test_suite.py",
         "tests/test_locomo.py", "tests/test_modelos.py", "tests/test_cartucho.py", "objetivos_nextlevel.txt"}
HEAVY_PREFIXES = ("experiment/baseline", "experiment/eval_", "experiment/audit_", "experiment/textos/")
MAX_BYTES = 400_000   # un archivo más grande se anuncia pero no se vuelca


def git(*args: str) -> str | None:
    try:
        p = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def excluded(rel: str) -> bool:
    parts = Path(rel).parts
    name = parts[-1]
    if any(p in EXCLUDED_DIRS for p in parts[:-1]) or name in EXCLUDED_FILES or name.endswith(EXCLUDED_SUFFIXES):
        return True
    if name.startswith("parche") and name.endswith(".py"):   # parches sueltos en la raíz
        return True
    if name.endswith(".egg-info") or any(p.endswith(".egg-info") for p in parts):
        return True
    return not (name in ALWAYS_INCLUDE or Path(name).suffix.lower() in TEXT_EXTENSIONS)


def candidate_files() -> tuple[list[str], str]:
    tracked = git("ls-files")
    if tracked is not None:
        new = git("ls-files", "--others", "--exclude-standard") or ""
        names = sorted(set(tracked.split("\n") + new.split("\n")) - {""})
        return [n for n in names if (PROJECT_ROOT / n).is_file()], "git (versionados + nuevos no ignorados)"
    names = sorted(str(p.relative_to(PROJECT_ROOT)) for p in PROJECT_ROOT.rglob("*") if p.is_file())
    return names, "recorrido de carpetas (sin git)"


def build_tree(files: list[str]) -> list[str]:
    tree: dict = {}
    for f in files:
        node = tree
        for part in Path(f).parts:
            node = node.setdefault(part, {})
    lines = [PROJECT_ROOT.name + "/"]

    def walk(node: dict, prefix: str) -> None:
        items = sorted(node.items(), key=lambda kv: (not kv[1], kv[0].lower()))
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            lines.append(prefix + ("└── " if last else "├── ") + name + ("/" if child else ""))
            if child:
                walk(child, prefix + ("    " if last else "│   "))

    walk(tree, "")
    return lines


def state_header() -> list[str]:
    out = []
    try:
        for line in (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version"):
                out.append("Versión: " + line.split("=", 1)[1].strip().strip('"'))
                break
    except OSError:
        pass
    desc = git("describe", "--tags", "--always", "--dirty")
    head = git("log", "-1", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M")
    dirty = [x for x in (git("status", "--porcelain") or "").splitlines() if x.strip()]
    if desc:
        out.append("Git: " + desc.strip() + (" · último commit: " + head.strip() if head else ""))
        out.append("Cambios sin commit: " + (", ".join(x[3:] for x in dirty[:15]) + (" …" if len(dirty) > 15 else "")
                                             if dirty else "ninguno"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Volcado del proyecto para Claude")
    ap.add_argument("--ligero", action="store_true", help="sin el banco, los tests, la consola ni los datos grandes")
    ap.add_argument("--solo", help="lista de archivos separados por comas (rutas relativas)")
    a = ap.parse_args()
    names, source = candidate_files()
    files = [n for n in names if not excluded(n)]
    if a.solo:
        wanted = {w.strip() for w in a.solo.split(",") if w.strip()}
        missing = wanted - set(files)
        files = [f for f in files if f in wanted]
        if missing:
            print("⚠ no encontrados o excluidos: " + ", ".join(sorted(missing)))
    elif a.ligero:
        files = [f for f in files if f not in HEAVY and not f.startswith(HEAVY_PREFIXES)]
    total = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        out.write("=" * 60 + "\nPROYECTO COMPLETO" + (" (ligero)" if a.ligero else " (parcial)" if a.solo else "") +
                  "\n" + "=" * 60 + "\n\n")
        out.write(f"Proyecto: {PROJECT_ROOT.name}\nGenerado: {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        out.write("\n".join(state_header()) + "\n")
        out.write(f"Archivos: {len(files)} · fuente: {source} · body/, .env, cachés y resultados excluidos siempre\n")
        if a.ligero:
            out.write("Omitidos por --ligero: " + ", ".join(sorted(HEAVY)) + ", experiment/{baseline,eval_,audit_,textos}*\n")
        out.write("\n" + "=" * 60 + "\nÁRBOL\n" + "=" * 60 + "\n\n" + "\n".join(build_tree(files)) + "\n\n")
        out.write("=" * 60 + "\nARCHIVOS\n" + "=" * 60 + "\n\n")
        for rel in files:
            path = PROJECT_ROOT / rel
            size = path.stat().st_size
            out.write("-" * 60 + f"\nFILE: {rel}\n" + "-" * 60 + "\n\n")
            if size > MAX_BYTES:
                out.write(f"[OMITIDO: {size / 1e3:.0f} KB; pídelo con --solo {rel}]\n\n")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                out.write("[OMITIDO: no es texto UTF-8]\n\n")
                continue
            except OSError as exc:
                out.write(f"[ERROR LEYENDO: {type(exc).__name__}: {exc}]\n\n")
                continue
            total += len(text)
            out.write(text + ("" if text.endswith("\n") else "\n") + "\n")
    kb = OUTPUT_FILE.stat().st_size / 1e3
    print(f"Proyecto exportado: {OUTPUT_FILE}")
    print(f"Archivos: {len(files)} · {kb:.0f} KB · ≈ {total // 4 / 1000:.0f}k tokens")
    if kb > 600 and not (a.ligero or a.solo):
        print("  (es grande: para conversar, prueba --ligero; para un parche, --solo con los archivos que toca)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())