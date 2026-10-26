#!/usr/bin/env python3
"""Evaluación de recuperación: delega en prueba.py (modo --live, solo categoría 'recuperacion')."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.exit(subprocess.call([sys.executable, str(root / "benchmarks" / "prueba.py"), "--live", "--only", "recuperacion", *sys.argv[1:]]))
