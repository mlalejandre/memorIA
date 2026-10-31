#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera los EDITS de un parche desde los cambios PREPARADOS (git add) de un repo: bloques mínimos y
únicos con contexto creciente, verificados aplicándolos en orden sobre HEAD hasta reproducir el árbol byte a byte.

    python3 scripts/generar_edits.py <repo> edits.json   → {"edits": [[archivo, viejo, nuevo, desc]], "new_files": {…}}
"""
import difflib, json, subprocess, sys
from pathlib import Path

repo = Path(sys.argv[1])
out = Path(sys.argv[2])


def git(*a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout


def find(cur, block):
    r = [l.rstrip() for l in block]
    rc = [l.rstrip() for l in cur]
    n = len(r)
    return [i for i in range(len(rc) - n + 1) if rc[i:i + n] == r]


def apply(text, old, new):
    cur = text.split("\n")
    o = old.split("\n")
    hits = find(cur, o)
    assert len(hits) == 1, len(hits)
    i = hits[0]
    return "\n".join(cur[:i] + new.split("\n") + cur[i + len(o):])


edits, new_files = [], {}
for line in git("diff", "--cached", "--name-status").splitlines():
    st, path = line.split("\t")
    if st == "A":
        new_files[path] = (repo / path).read_text(encoding="utf-8")
        continue
    old_text = git("show", f"HEAD:{path}")
    new_text = (repo / path).read_text(encoding="utf-8")
    cur = old_text.split("\n")
    tgt = new_text.split("\n")
    n_edit = 0
    while cur != tgt:
        sm = difflib.SequenceMatcher(None, cur, tgt, autojunk=False)
        ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
        tag, i1, i2, j1, j2 = ops[0]
        a = b = 0
        # contexto disponible: líneas iguales antes/después
        while True:
            oldb = cur[i1 - a:i2 + b]
            if oldb and any(x.strip() for x in oldb) and len(find(cur, oldb)) == 1:
                break
            if a <= b and i1 - a > 0 and cur[i1 - a - 1] == tgt[j1 - a - 1]:
                a += 1
            elif i2 + b < len(cur) and j2 + b < len(tgt) and cur[i2 + b] == tgt[j2 + b]:
                b += 1
            elif i1 - a > 0 and cur[i1 - a - 1] == tgt[j1 - a - 1]:
                a += 1
            else:
                raise SystemExit(f"sin contexto único en {path}")
        old = "\n".join(cur[i1 - a:i2 + b])
        new = "\n".join(tgt[j1 - a:j2 + b])
        n_edit += 1
        edits.append([path, old, new, f"{path} · cambio {n_edit}"])
        cur = apply("\n".join(cur), old, new).split("\n")

# verificación: aplicar en orden sobre HEAD reproduce el árbol byte a byte
state = {}
for path, old, new, _ in edits:
    if path not in state:
        state[path] = git("show", f"HEAD:{path}")
    state[path] = apply(state[path], old, new)
for path, text in state.items():
    assert text == (repo / path).read_text(encoding="utf-8"), path
out.write_text(json.dumps({"edits": edits, "new_files": new_files}, ensure_ascii=False), encoding="utf-8")
print(f"{len(edits)} edits en {len(state)} archivos · {len(new_files)} nuevos · verificado byte a byte")
