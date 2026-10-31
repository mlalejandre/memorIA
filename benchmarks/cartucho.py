# -*- coding: utf-8 -*-
"""Examen de un cartucho con preguntas escritas a mano (documento cargado por el usuario).

    python3 benchmarks/cartucho.py --cartucho santander --preguntas experiment/preguntas_santander.json

· Trabaja sobre una COPIA de body/cartuchos/<id>/memoria.db (copia consistente con la API de backup de SQLite):
  el cartucho original solo se lee.
· Modos: «sin memoria» (lo que el modelo ya sabe), y con memoria · pensar no / auto / siempre.
· Mismo banco que LoCoMo: respuestas con T=0, una a la vez y sin caché de prompt; juez de un voto con T=0 en el
  servidor de la memoria larga.
· Por cada pregunta, «claves» (fragmentos que debe contener la evidencia) separan los fallos en: ingesta (la
  memoria no tiene el dato), recuperación (está en la memoria pero no llega al prompt) y respuesta (llegó y falla).

Desde textos, sin tocar ningún cartucho:
    python3 benchmarks/cartucho.py --textos "experiment/textos/cabo_machichaco.txt=Wikipedia: Cabo Machichaco (vapor);\
experiment/textos/incendio_santander_1941.txt=Wikipedia: Incendio de Santander de 1941" \
        --variantes actual,autocontenidos --preguntas experiment/preguntas_santander.json --modos no
  Ingiere cada texto con SU fuente en una memoria temporal por variante (actual = extract_facts de siempre;
  autocontenidos = src/documentos.py) y compara las variantes pregunta a pregunta.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT, ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("HMEM_FIXED_NOW", "31/10/2026 12:00")

import config as C  # noqa: E402

MODES = ("sin memoria", "no", "auto", "siempre", "atrib", "ref")   # atrib = no + ATTRIB_EVENTS · ref = como «no», pero responde el servidor LTM

POS_JUDGE = (
    "Eres un evaluador. Se hizo a un asistente con memoria una pregunta sobre unos documentos que había guardado, y "
    "existe una RESPUESTA DE REFERENCIA. Decide si la respuesta del asistente es correcta: contiene lo esencial de la "
    "referencia (aunque use otras palabras o añada detalles correctos) y no la contradice. Las cifras y las fechas deben "
    "coincidir con la referencia. Si la referencia tiene varios elementos, basta con que acierte la mayoría. Si dice "
    "que no lo sabe, o da como respuesta principal otro dato, es incorrecta.\n"
    'Devuelve SOLO JSON: {"correcta": true} o {"correcta": false}.')
NEG_JUDGE = (
    "Eres un evaluador. Se hizo a un asistente con memoria una pregunta TRAMPA sobre unos documentos que había guardado: "
    "da por hecho algo que los documentos no dicen (a menudo mezcla datos de dos sucesos distintos). Decide si la "
    "respuesta INVENTA: afirma como cierto un dato concreto que responde a la premisa falsa (la respuesta trampa "
    "indicada u otro valor). NO es inventar: decir que no consta, o corregir la premisa (por ejemplo, señalar que ese "
    "dato corresponde al otro suceso).\n"
    'Devuelve SOLO JSON: {"inventa": true} o {"inventa": false}.')


def norm(s: Any) -> str:
    """minúsculas, sin tildes y sin separadores de miles (para buscar claves)."""
    t = unicodedata.normalize("NFKD", str(s or "").lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.replace(" ", " ").replace(".", "").replace(" 000", "000")


def has_keys(text: str, keys: list[str]) -> bool:
    """True si el texto contiene TODAS las claves (cada clave admite alternativas separadas por «|»)."""
    t = norm(text)
    return all(any(norm(alt) in t for alt in str(k).split("|")) for k in keys) if keys else False


def cause(in_memory: bool, in_prompt: bool) -> str:
    return "respuesta" if in_prompt else ("recuperación" if in_memory else "ingesta")


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    qs = data["preguntas"] if isinstance(data, dict) else data
    out = []
    for i, q in enumerate(qs, 1):
        q = dict(q)
        q.setdefault("id", f"q{i}")
        q.setdefault("tipo", "directa")
        q.setdefault("claves", [])
        if q["tipo"] == "trampa":
            q.setdefault("trampa", "")
        out.append(q)
    return out


def copy_cartridge(cid: str, dst: Path) -> Path:
    """Copia consistente de memoria.db del cartucho (solo lectura del original)."""
    src_dir = Path(getattr(C, "CARTRIDGES_DIR", C.BODY / "cartuchos")) / cid
    src = src_dir / "memoria.db"
    if not src.exists():
        raise FileNotFoundError(f"no existe {src}")
    dst.mkdir(parents=True, exist_ok=True)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30.0)
    d = sqlite3.connect(str(dst / "memoria.db"))
    with d:
        s.backup(d)
    s.close()
    d.close()
    if (src_dir / "manifest.json").exists():
        shutil.copy2(src_dir / "manifest.json", dst / "manifest.json")
    return dst


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumen por modo: correctas (no trampa), inventa (trampa), evidencia, causas, tiempo."""
    out: dict[str, Any] = {}
    for mode in MODES:
        rs = [r for r in rows if r["modo"] == mode]
        if not rs:
            continue
        pos = [r for r in rs if r["tipo"] != "trampa"]
        neg = [r for r in rs if r["tipo"] == "trampa"]
        ok = [r for r in pos if r["veredicto"] is True]
        inv = [r for r in neg if r["veredicto"] is not False]
        bad = [r for r in pos if r["veredicto"] is not True]
        causes: dict[str, int] = {}
        if mode != "sin memoria":
            for r in bad:
                causes[r["causa"]] = causes.get(r["causa"], 0) + 1
        by_type: dict[str, list[int]] = {}
        for r in pos:
            by_type.setdefault(r["tipo"], [0, 0])
            by_type[r["tipo"]][0] += r["veredicto"] is True
            by_type[r["tipo"]][1] += 1
        ts = sorted(r["t"] for r in rs)
        out[mode] = {"n_pos": len(pos), "n_neg": len(neg), "correctas": len(ok), "inventa": len(inv),
                     "causas": causes, "por_tipo": by_type, "t_p50": ts[len(ts) // 2] if ts else 0.0,
                     "t_p95": ts[int(0.95 * (len(ts) - 1))] if ts else 0.0,
                     "piensan": sum(1 for r in rs if r.get("piensa"))}
    return out


def paired(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    """Comparación pareada a → b (correcta en positivas, no inventar en trampas)."""
    from locomo import sign_p
    good = lambda r: (r["veredicto"] is True) if r["tipo"] != "trampa" else (r["veredicto"] is False)
    ra = {r["id"]: r for r in rows if r["modo"] == a}
    rb = {r["id"]: r for r in rows if r["modo"] == b}
    ids = [i for i in ra if i in rb]
    up = sum(1 for i in ids if not good(ra[i]) and good(rb[i]))
    down = sum(1 for i in ids if good(ra[i]) and not good(rb[i]))
    out = {"n": len(ids), "mejoran": up, "empeoran": down, "p": sign_p(down, up)}
    for name, sel in (("positivas", lambda r: r["tipo"] != "trampa"), ("trampas", lambda r: r["tipo"] == "trampa")):
        ii = [i for i in ids if sel(ra[i])]   # desglose
        u = sum(1 for i in ii if not good(ra[i]) and good(rb[i]))
        d = sum(1 for i in ii if good(ra[i]) and not good(rb[i]))
        out[name] = {"n": len(ii), "mejoran": u, "empeoran": d, "p": sign_p(d, u)}
    return out


VARIANTS = ("actual", "actual2", "autocontenidos")   # actual2 = otra ingesta igual: ruido de la ingesta


def paired_variants(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], mode: str = "no") -> dict[str, Any]:
    """Variante a → b en el mismo modo (correcta en positivas, no inventar en trampas)."""
    from locomo import sign_p
    good = lambda r: (r["veredicto"] is True) if r["tipo"] != "trampa" else (r["veredicto"] is False)
    ra = {r["id"]: r for r in rows_a if r["modo"] == mode}
    rb = {r["id"]: r for r in rows_b if r["modo"] == mode}
    ids = [i for i in ra if i in rb]
    out = {}
    for name, sel in (("todas", lambda r: True), ("positivas", lambda r: r["tipo"] != "trampa"),
                      ("trampas", lambda r: r["tipo"] == "trampa")):
        ii = [i for i in ids if sel(ra[i])]
        up = sum(1 for i in ii if not good(ra[i]) and good(rb[i]))
        down = sum(1 for i in ii if good(ra[i]) and not good(rb[i]))
        out[name] = {"n": len(ii), "mejoran": up, "empeoran": down, "p": sign_p(down, up)}
    return out


def parse_texts(spec: str) -> list[tuple[Path, str]]:
    out = []
    for part in [x for x in spec.split(";") if x.strip()]:
        path, _, src = part.partition("=")
        out.append((Path(path.strip()), src.strip() or Path(path.strip()).stem))
    return out


def build_from_texts(dest: Path, texts: list[tuple[Path, str]], variant: str) -> dict[str, Any]:
    """Ingesta REAL (como el panel de la consola) de cada texto con SU fuente en una memoria nueva."""
    from agente import split_chunks
    from largo_plazo import MemLLM, MemoryTree
    import documentos
    t0 = time.time()
    tree = MemoryTree(dest, MemLLM())
    n_facts = n_fail = n_chunks = 0
    for path, src in texts:
        chunks = split_chunks(Path(path).read_text(encoding="utf-8"))
        n_chunks += len(chunks)
        for idx, chunk in enumerate(chunks):
            facts = None
            if variant == "autocontenidos":
                facts = documentos.extract(tree.llm, chunk, src, chunks[idx - 1] if idx > 0 else "")
            if facts is None:
                _det, facts = tree.extract_facts(chunk, first=False)
            if facts is None:
                n_fail += 1
                continue
            for f in facts:
                imp = max(9.0, f["importancia"]) if f["tipo"] == "instruccion" else f["importancia"]
                tree.store(f["texto"], imp, f["tipo"], meta={"fuente": src})
                n_facts += 1
        print(f"   · [{variant}] {Path(path).name}: {len(chunks)} bloques · {n_facts} hechos hasta ahora", flush=True)
    return {"tree": tree, "hechos_guardados": n_facts, "bloques": n_chunks, "fallos_extraccion": n_fail,
            "min_ingesta": (time.time() - t0) / 60}


def exam(tree: Any, label: str, qs: list[dict[str, Any]], modes: list[str], chat: Any, jchat: Any) -> dict[str, Any]:
    from agente import BASE_PROMPT, Agent, now_str
    import pensar
    from locomo import fact_lines, judge_once
    t_start = time.time()
    cells = [str(r[0]) for r in tree.conn.execute(
        "SELECT text FROM cells WHERE archived = 0 AND valid_until IS NULL AND superseded_by IS NULL")]
    n_nodes = int(tree.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
    try:
        n_epi = int(tree.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
    except sqlite3.Error:
        n_epi = 0
    all_text = "\n".join(cells)
    print(f"\n   [{label}] {len(cells)} hechos vigentes · {n_nodes} nodos · {n_epi} episodios")
    ret: dict[str, dict[str, Any]] = {}
    for q in qs:
        t0 = time.perf_counter()
        r = tree.recall(q["q"], reinforce=False, synthesize=False)
        epi = tree.search_episodes(q["q"], int(getattr(C, "EPI_LOOKUP_K", 22))) or []
        ms = (time.perf_counter() - t0) * 1000
        found, tent, ans = bool(r.get("found")), bool(r.get("tentative")), str(r.get("answer") or "")
        ltm = {"found": found, "tentative": tent, "sources": list(r.get("sources") or []),
               "answer": "" if tent else (ans if found else ""), "tentative_answer": ans if (found and tent) else ""}
        system = Agent._build_system(None, {}, ltm, epi)
        lines = fact_lines(ans) if found else []
        n_epi_prompt = max(1, int(getattr(C, "EPI_PROMPT_K", 5)))
        ctx_text = "\n".join(lines) + "\n" + "\n".join(str(e.get("text", "")) for e in epi[:n_epi_prompt])
        ret[q["id"]] = {"system": system, "ms": ms, "hechos": lines, "ltm": ltm, "epi": epi,
                        "en_memoria": has_keys(all_text, q["claves"]), "en_prompt": has_keys(ctx_text, q["claves"])}
    lat = sorted(v["ms"] for v in ret.values())
    p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0
    pos_q = [q for q in qs if q["tipo"] != "trampa"]
    print(f"   [{label}] recuperación: P95 {p95:.0f} ms · el dato está en la memoria en "
          f"{sum(ret[q['id']]['en_memoria'] for q in pos_q)}/{len(pos_q)} y llega al prompt en "
          f"{sum(ret[q['id']]['en_prompt'] for q in pos_q)}/{len(pos_q)} (según las claves)")
    rows: list[dict[str, Any]] = []
    jobs = [(m, q) for m in modes for q in qs]

    def judge(row: dict[str, Any]) -> None:
        if row["tipo"] == "trampa":
            user = f"Pregunta: {row['q']}\nRespuesta trampa: {row['trampa']}\nRespuesta del asistente: {row['respuesta']}"
            row["veredicto"] = judge_once(jchat, NEG_JUDGE, user, "inventa")
        else:
            user = f"Pregunta: {row['q']}\nRespuesta de referencia: {row['ref']}\nRespuesta del asistente: {row['respuesta']}"
            row["veredicto"] = judge_once(jchat, POS_JUDGE, user, "correcta")

    print(f"   [{label}] … {len(jobs)} respuestas (1 a la vez, T=0) + juez a la vez · modos: {', '.join(modes)}", flush=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exj:
        futs = []
        for k, (mode, q) in enumerate(jobs, 1):
            if mode == "sin memoria":
                system, th, why = BASE_PROMPT.format(now=now_str()), False, "sin memoria"
            elif mode == "ref":   # mismo prompt que «no», respondido por el servidor de la memoria larga
                system, th, why = ret[q["id"]]["system"], False, "modelo de referencia (LTM)"
            elif mode == "atrib":   # mismo contexto, con la regla de sucesos y documentos
                prev = getattr(C, "ATTRIB_EVENTS", False)
                C.ATTRIB_EVENTS = True
                try:
                    system = Agent._build_system(None, {}, ret[q["id"]]["ltm"], ret[q["id"]]["epi"])
                finally:
                    C.ATTRIB_EVENTS = prev
                th, why = False, "regla de sucesos"
            else:
                system = ret[q["id"]]["system"]
                th, why = pensar.decide(q["q"], {"no": "off", "auto": "auto", "siempre": "on"}[mode])
            mt = int(C.ANSWER_MAX_TOKENS) + pensar.extra_tokens() if th else 700
            t_a = time.perf_counter()
            try:
                who = jchat if mode == "ref" else chat
                ans = who.chat([{"role": "system", "content": system}, {"role": "user", "content": q["q"]}],
                               max_tokens=mt, temperature=0.0, cache_prompt=False,
                               **({"thinking": True} if th else {"thinking": False}))
            except Exception as exc:
                ans = f"[error LLM: {type(exc).__name__}]"
            info = ret[q["id"]]
            row = {"id": q["id"], "modo": mode, "tipo": q["tipo"], "q": q["q"], "ref": q.get("ref", ""),
                   "trampa": q.get("trampa", ""), "respuesta": ans, "t": round(time.perf_counter() - t_a, 2),
                   "piensa": th, "motivo_piensa": why, "en_memoria": info["en_memoria"],
                   "en_prompt": info["en_prompt"], "causa": cause(info["en_memoria"], info["en_prompt"]),
                   "hechos": info["hechos"][:15]}
            rows.append(row)
            futs.append(exj.submit(judge, row))
            if k % 20 == 0 or k == len(jobs):
                el = (time.time() - t0) / 60
                print(f"     {k}/{len(jobs)} respuestas · {el:.0f} min · quedan ≈{el / k * (len(jobs) - k):.0f} min",
                      flush=True)
        for f in futs:
            f.result()
    summ = summarize(rows)
    print(f"\n   RESULTADOS [{label}]")
    for mode, s in summ.items():
        causes = " · ".join(f"{k} {v}" for k, v in sorted(s["causas"].items(), key=lambda x: -x[1])) or "—"
        types = " · ".join(f"{t} {v[0]}/{v[1]}" for t, v in sorted(s["por_tipo"].items()))
        print(f"   {mode:<12} correctas {s['correctas']}/{s['n_pos']} = {100 * s['correctas'] / max(1, s['n_pos']):.1f} % · "
              f"inventa {s['inventa']}/{s['n_neg']} · mediana {s['t_p50']:.1f} s · piensan {s['piensan']} · "
              f"fallos: {causes}")
        print(f"   {'':<12} por tipo: {types}")
    pairs = {f"{x}→{y}": paired(rows, x, y) for x, y in (("sin memoria", "no"), ("no", "auto"), ("no", "siempre"),
                                                          ("no", "atrib"), ("ref", "no")) if x in modes and y in modes}
    for k, v in pairs.items():
        print(f"   pareado {k}: n={v['n']} · mejoran {v['mejoran']} · empeoran {v['empeoran']} · p={v['p']:.3f}")
    srcs: dict[str, int] = {}
    for r in tree.conn.execute("SELECT fuente FROM cells WHERE archived = 0"):
        srcs[str(r[0])] = srcs.get(str(r[0]), 0) + 1
    return {"etiqueta": label, "hechos": len(cells), "nodos": n_nodes, "episodios": n_epi, "p95_ms": p95,
            "fuentes": srcs, "resumen": summ, "pareados": pairs, "min": (time.time() - t_start) / 60,
            "evidencia": {"en_memoria": sum(ret[q['id']]['en_memoria'] for q in pos_q),
                          "en_prompt": sum(ret[q['id']]['en_prompt'] for q in pos_q), "n": len(pos_q)},
            "filas": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="Examen de un cartucho (o de textos ingeridos aparte) con preguntas a mano")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cartucho")
    g.add_argument("--textos", help='"ruta=fuente;ruta=fuente" · se ingieren en memorias temporales')
    ap.add_argument("--variantes", default=",".join(VARIANTS))
    ap.add_argument("--preguntas", required=True)
    ap.add_argument("--modos", default=",".join(MODES))
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    from largo_plazo import MemLLM, MemoryTree
    from llm import LLM

    qs = load_questions(Path(a.preguntas))
    modes = [m.strip() for m in a.modos.split(",") if m.strip() in MODES]
    tmp = Path(tempfile.mkdtemp(prefix="hmem_cartucho_"))
    t_start = time.time()
    chat = LLM(base_url=C.LLM_STM_URL, model=C.REMOTE_MODEL_ALIAS)
    jchat = LLM(base_url=os.getenv("HMEM_JUDGE_URL", "").strip() or C.LLM_LTM_URL, model=C.REMOTE_MODEL_ALIAS)
    print(f"   {len(qs)} preguntas: " + " · ".join(
        f"{t} {sum(1 for q in qs if q['tipo'] == t)}" for t in sorted({q['tipo'] for q in qs})))
    trees = []
    try:
        if a.cartucho:
            try:
                copy_cartridge(a.cartucho, tmp / a.cartucho)
            except FileNotFoundError as exc:
                print(f"❌ Cartucho «{a.cartucho}» no encontrado ({exc}). Mira los nombres en body/cartuchos/.")
                return 1
            tree = MemoryTree(tmp / a.cartucho, MemLLM())
            trees.append(tree)
            out = exam(tree, f"cartucho {a.cartucho} (copia)", qs, modes, chat, jchat)
            out["cartucho"] = a.cartucho
        else:
            texts = parse_texts(a.textos)
            for p, _s in texts:
                if not p.exists():
                    print(f"❌ No existe {p}")
                    return 1
            variants = [v.strip() for v in a.variantes.split(",") if v.strip() in VARIANTS]
            out = {"textos": [[str(p), s] for p, s in texts], "variantes": {}}
            for v in variants:
                print(f"\n   … ingesta [{v}] de {len(texts)} textos (cada uno con su fuente)", flush=True)
                b = build_from_texts(tmp / v, texts, v)
                trees.append(b["tree"])
                print(f"   [{v}] ingesta: {b['hechos_guardados']} hechos de {b['bloques']} bloques en "
                      f"{b['min_ingesta']:.1f} min · fallos de extracción {b['fallos_extraccion']}", flush=True)
                e = exam(b["tree"], v, qs, modes, chat, jchat)
                e.update({k: b[k] for k in ("hechos_guardados", "bloques", "fallos_extraccion", "min_ingesta")})
                out["variantes"][v] = e
            out["pareado"] = {}
            for b2 in ("actual2", "autocontenidos"):
                if "actual" in out["variantes"] and b2 in out["variantes"]:
                    out["pareado"][f"actual→{b2}"] = {m: paired_variants(out["variantes"]["actual"]["filas"],
                                                                         out["variantes"][b2]["filas"], m) for m in modes}
            for name, per_mode in out["pareado"].items():
                for m, pv in per_mode.items():
                    for k, v in pv.items():
                        print(f"   pareado {name} · {m} · {k}: n={v['n']} · mejoran {v['mejoran']} · "
                              f"empeoran {v['empeoran']} · p={v['p']:.3f}")
        out.update({"fecha": time.strftime("%Y-%m-%dT%H:%M:%S"), "min": (time.time() - t_start) / 60})
        name = a.cartucho or "textos"
        dest = Path(a.json) if a.json else ROOT / "resultados_parches" / f"cartucho_{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n   detalle: {dest}")
        return 0
    finally:
        for tr in trees:
            try:
                tr.close()
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
