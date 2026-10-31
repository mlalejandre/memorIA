from __future__ import annotations

from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ahora en scripts/
OUTPUT_FILE = PROJECT_ROOT / "proyecto_completo.txt"

# Directorios que NO queremos exportar.
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "logs",
    "snapshots",
    "autoprompt_history",
    "backups",
    ".parche3_backup",
    ".parche_backup",
    "resultados_parches",
    "locomo_cache",  # datos de LoCoMo y memorias ingeridas
}

# Archivos que NO queremos incluir.
EXCLUDED_FILES = {
    OUTPUT_FILE.name,
    ".DS_Store",
    "parche.py",
    "parche2.py",
    "fix_parche.py",
    ".env",
    "sesion.json",
    "_ltm_pendiente.jsonl",
    "_archivo.jsonl",
}

# Sufijos de copias de seguridad / residuos que no deben entrar en el volcado.
EXCLUDED_SUFFIXES = {
    ".bak",
    ".pre_parche",
    ".tmp",
}

# Extensiones que consideramos código/texto relevante.
TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".sh",
    ".bash",
    ".zsh",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
}

# Archivos concretos que siempre incluiremos aunque no tengan
# una de las extensiones anteriores.
ALWAYS_INCLUDE = {
    "Dockerfile",
    "Makefile",
    "requirements.txt",
}


# ============================================================
# UTILIDADES
# ============================================================

def is_excluded_dir(path: Path) -> bool:
    return path.name in EXCLUDED_DIRS


def is_excluded_file(path: Path) -> bool:
    if path.name in EXCLUDED_FILES:
        return True
    # Excluir residuos de parches y backups (memoria.py.bak, *.pre_parche, …)
    name = path.name
    for suf in EXCLUDED_SUFFIXES:
        if name.endswith(suf):
            return True
    if name.startswith("parche") and name.endswith(".py"):  # scripts de parche
        return True
    return False


def is_text_file(path: Path) -> bool:
    return (
        path.name in ALWAYS_INCLUDE
        or path.suffix.lower() in TEXT_EXTENSIONS
    )


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


# ============================================================
# ÁRBOL DE DIRECTORIOS
# ============================================================

def build_tree(root: Path) -> list[str]:
    lines: list[str] = []

    def walk(directory: Path, prefix: str = "") -> None:
        entries = []

        for path in sorted(directory.iterdir(), key=lambda p: (
            not p.is_dir(),
            p.name.lower(),
        )):
            if path.is_dir():
                if is_excluded_dir(path):
                    continue
            else:
                if is_excluded_file(path):
                    continue

            entries.append(path)

        for index, path in enumerate(entries):
            is_last = index == len(entries) - 1
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + path.name)

            if path.is_dir():
                extension = "    " if is_last else "│   "
                walk(path, prefix + extension)

    lines.append(PROJECT_ROOT.name + "/")
    walk(root)

    return lines


# ============================================================
# CONTENIDO DE ARCHIVOS
# ============================================================

def collect_source_files(root: Path) -> list[Path]:
    files: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        relative_parts = path.relative_to(root).parts

        if any(part in EXCLUDED_DIRS for part in relative_parts):
            continue

        if is_excluded_file(path):
            continue

        if not is_text_file(path):
            continue

        files.append(path)

    return files


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return (
            f"[ERROR LEYENDO ARCHIVO: "
            f"{type(exc).__name__}: {exc}]"
        )


# ============================================================
# EXPORTACIÓN
# ============================================================

def export_project() -> None:
    tree = build_tree(PROJECT_ROOT)
    source_files = collect_source_files(PROJECT_ROOT)

    generated_at = datetime.now().astimezone().isoformat()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        output.write(
            "============================================================\n"
            "PROYECTO COMPLETO\n"
            "============================================================\n\n"
        )

        output.write(
            f"Ruta del proyecto: {PROJECT_ROOT}\n"
            f"Generado: {generated_at}\n\n"
        )

        output.write(
            "============================================================\n"
            "ÁRBOL DE DIRECTORIOS\n"
            "============================================================\n\n"
        )

        output.write("\n".join(tree))
        output.write("\n\n")

        output.write(
            "============================================================\n"
            "ARCHIVOS Y CÓDIGO\n"
            "============================================================\n\n"
        )

        for path in source_files:
            rel = relative(path)
            content = read_text_file(path)

            output.write(
                "------------------------------------------------------------\n"
            )
            output.write(f"FILE: {rel}\n")
            output.write(
                "------------------------------------------------------------\n\n"
            )

            output.write(content)

            if not content.endswith("\n"):
                output.write("\n")

            output.write("\n")

    print(
        f"Proyecto exportado correctamente:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Archivos incluidos: {len(source_files)}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    export_project()
