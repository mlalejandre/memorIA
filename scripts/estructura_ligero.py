#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Volcado LIGERO del proyecto (lo mismo que «python3 scripts/estructura.py --ligero»).

    python3 scripts/estructura_ligero.py

Genera proyecto_completo.txt sin el banco de pruebas, los tests, la consola web ni los datos grandes de
experiment/: es el que conviene adjuntar, junto con resumen.txt, para retomar el trabajo con Claude.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import estructura  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--ligero", *sys.argv[1:]]
    raise SystemExit(estructura.main())