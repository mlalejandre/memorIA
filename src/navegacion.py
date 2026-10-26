# Navegación por el árbol semántico (beam search guiado por LLM).
from __future__ import annotations

import re
import unicodedata
from typing import Any

import config as C

ROUTER_SYSTEM = (
    "Eres el enrutador de una memoria jerárquica en forma de árbol (de lo general a lo concreto). "
    "Responde SIEMPRE y SOLO con un JSON válido, sin texto adicional."
)


def enabled() -> bool:
    return bool(getattr(C, "MEM_NAV", True))


def _fold(s: Any) -> str:
    t = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _stem(w: str) -> str:
    w = _fold(w)
    return w[: max(4, len(w) - 3)] if len(w) >= 6 else w


def _stems(tree: Any, text: str) -> set[str]:
    return {_stem(t) for t in tree._fts_terms(text, limit=12)}


def _overlap(text: str, qstems: set[str]) -> int:
    if not qstems:
        return 0
    toks = re.findall(r"[^\W_]+", _fold(text))
    return sum(1 for q in qstems if any(t.startswith(q) for t in toks))


def _graph(tree: Any, include_history: bool = False):
    with tree.lock:
        cur = tree.conn.cursor()
        cur.execute("SELECT id, parent_id, name, description FROM nodes")
        nodes = {r["id"]: {"id": r["id"], "parent": r["parent_id"],
                           "name": r["name"] or "", "desc": r["description"] or ""}
                 for r in cur.fetchall()}
        temporal_filter = "" if include_history else " WHERE valid_until IS NULL AND superseded_by IS NULL"
        cur.execute("SELECT node_id, COUNT(*) AS c FROM cells" + temporal_filter + " GROUP BY node_id")
        own = {r["node_id"]: r["c"] for r in cur.fetchall()}
    kids: dict[Any, list[int]] = {}
    for n in nodes.values():
        kids.setdefault(n["parent"], []).append(n["id"])
    return nodes, kids, own


def _totals(nodes: dict, kids: dict, base: dict) -> dict[int, int]:
    total: dict[int, int] = {}

    def rec(nid: int) -> int:
        if nid in total:
            return total[nid]
        total[nid] = 0  # protege frente a ciclos
        t = base.get(nid, 0) + sum(rec(c) for c in kids.get(nid, []))
        total[nid] = t
        return t

    for nid in nodes:
        rec(nid)
    return total


def _path(nodes: dict, nid: int) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    cur = nid
    while cur is not None and cur in nodes and cur not in seen:
        seen.add(cur)
        parts.append(nodes[cur]["name"])
        cur = nodes[cur]["parent"]
    return " › ".join(reversed(parts))


def _choose(tree, question, frontier, nodes, total, hits, qst, beam, list_limit, may_call, node_sims=None):
    """Devuelve (ids_elegidos, llamadas_LLM_usadas).
    Navegación vectorial/léxica Zero-LLM por defecto."""
    sims = node_sims or {}

    def score(nid: int) -> float:
        n = nodes[nid]
        ov = _overlap(n["name"] + " " + n["desc"], qst)
        ch_hit = hits.get(nid, 0)
        s = sims.get(nid, 0.0) if sims else 0.0
        return (s * 3.0) + (ov * 1.5) + (min(ch_hit, 4) * 0.75)

    ranked = sorted(frontier, key=lambda i: (-score(i), -total.get(i, 0)))
    shown = ranked[:list_limit]

    # VÍA RÁPIDA: Si hay vectores activos y no se fuerza MEM_NAV_LLM=1, seleccionamos directamente
    # las mejores ramas vectorialmente sin llamar al LLM (latencia < 2 ms)
    allow_llm = bool(getattr(C, "MEM_NAV_LLM", False))
    if sims and any(sims.get(i, -1.0) > 0.0 for i in frontier) and not allow_llm:
        chosen = [i for i in ranked if score(i) > 0.10][:beam]
        return (chosen or ranked[:1]), 0

    # VÍA LLM (solo como fallback si los embeddings están apagados o se configuró MEM_NAV_LLM=1)
    if may_call:
        lines = []
        for idx, nid in enumerate(shown, 1):
            n = nodes[nid]
            extra = f" · {hits[nid]} coincidencias de palabras" if hits.get(nid) else ""
            if nid in sims:
                extra += f" · similitud semántica {sims[nid]:.2f}"
            lines.append(f"[{idx}] {_path(nodes, nid)} — {n['desc'] or 'sin descripción'} "
                         f"({total.get(nid, 0)} recuerdos{extra})")
        user = (
            "Pregunta del usuario: " + question + "\n"
            "Navegas un árbol de memoria de lo general a lo concreto. Estas son las ramas candidatas:\n"
            + "\n".join(lines) + "\n"
            f"Elige hasta {beam} ramas donde sea más probable que estén los recuerdos que responden a la "
            "pregunta, aunque la pregunta use otras palabras (p. ej. 'mascota' puede estar en 'perro'). "
            "Si ninguna es relevante, devuelve una lista vacía.\n"
            'Devuelve JSON: {"elegidos": [1]}'
        )
        try:
            obj = tree.llm.ask_json(ROUTER_SYSTEM, user)
        except Exception:
            obj = None
        picks = obj.get("elegidos") if isinstance(obj, dict) else None
        if isinstance(picks, list):
            chosen: list[int] = []
            for p in picks:
                try:
                    i = int(p)
                except (TypeError, ValueError):
                    continue
                if 1 <= i <= len(shown) and shown[i - 1] not in chosen:
                    chosen.append(shown[i - 1])
            return chosen[:beam], 1
        used = 1
    else:
        used = 0

    # Heurística de reserva cuando no hay LLM ni vectores dominantes
    positives = [i for i in ranked if score(i) > 0.0][:beam]
    return (positives or ranked[:1]), used


def _node_sims(tree: Any, question: str) -> dict[int, float]:
    emb = getattr(tree, "emb", None)
    if emb is None:
        return {}
    try:
        return emb.node_similarities(question)
    except Exception:
        return {}


def navigate(tree: Any, question: str, beam: int | None = None, max_cells: int = 20,
            include_history: bool = False) -> list[dict[str, Any]]:
    """Baja por el árbol eligiendo las `beam` mejores ramas por nivel y devuelve células candidatas."""
    question = (question or "").strip()
    if not question:
        return []
    beam = max(1, int(beam or getattr(C, "MEM_BEAM", 3)))
    max_calls = max(1, int(getattr(C, "MEM_NAV_MAX_CALLS", 40)))
    list_limit = max(4, int(getattr(C, "MEM_LIST_LIMIT", 30)))

    nodes, kids, own = _graph(tree, include_history=include_history)
    if not nodes:
        return []
    qst = _stems(tree, question)
    hit_base: dict[int, int] = {}
    for c in tree._fts_search(tree._fts_terms(question), limit=30):
        hit_base[c["node_id"]] = hit_base.get(c["node_id"], 0) + 1
    total = _totals(nodes, kids, own)
    hits = _totals(nodes, kids, hit_base)
    node_sims = _node_sims(tree, question)

    frontier = [i for i in kids.get(None, []) if total.get(i, 0) > 0]
    collect: list[int] = []
    calls = 0
    for _ in range(16):
        if not frontier:
            break
        if len(frontier) == 1:
            chosen = list(frontier)
        else:
            allow_llm = bool(getattr(C, 'MEM_NAV_LLM', False)) or (not node_sims)
            chosen, used = _choose(tree, question, frontier, nodes, total, hits, qst,
                                   beam, list_limit, (calls < max_calls) and allow_llm, node_sims)
            calls += used
        nxt: list[int] = []
        for nid in chosen:
            if own.get(nid):
                collect.append(nid)
            nxt.extend(c for c in kids.get(nid, []) if total.get(c, 0) > 0)
        frontier = nxt

    if not collect:
        return []
    marks = ",".join("?" * len(collect))
    with tree.lock:
        cur = tree.conn.cursor()
        temporal_filter = "" if include_history else "AND valid_until IS NULL AND superseded_by IS NULL "
        cur.execute(f"SELECT * FROM cells WHERE node_id IN ({marks}) " + temporal_filter +
                    "ORDER BY archived ASC, importance DESC LIMIT 400", collect)
        cells = [dict(r) for r in cur.fetchall()]

    # Si hay embeddings disponibles, ordenamos las células candidatas por similitud semántica
    emb = getattr(tree, "emb", None)
    if emb is not None and len(cells) > 1:
        try:
            qv = emb.embed_query(question)
            if qv is not None:
                sim_map = dict(emb.store.search("cell", qv, len(cells) + 5, -1.0, emb.model))
                for c in cells:
                    c["sim"] = sim_map.get(c["id"], c.get("sim", 0.0))
                cells.sort(key=lambda c: (int(c.get("archived") or 0), -float(c.get("sim", 0.0)),
                                          -_overlap(c["text"], qst), -float(c["importance"])))
            else:
                cells.sort(key=lambda c: (int(c.get("archived") or 0), -_overlap(c["text"], qst),
                                          -float(c["importance"])))
        except Exception:
            cells.sort(key=lambda c: (int(c.get("archived") or 0), -_overlap(c["text"], qst),
                                      -float(c["importance"])))
    else:
        cells.sort(key=lambda c: (int(c.get("archived") or 0), -_overlap(c["text"], qst),
                                  -float(c["importance"])))

    return cells[:max_cells]


