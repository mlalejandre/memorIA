# -*- coding: utf-8 -*-
# locomo.py — Benchmark PÚBLICO LoCoMo (Maharana et al., ACL 2024) medido contra el OBJETIVO.  [PARCHE10]
#
# Por qué: hasta v0.3.0a1 todo se midió con un examen escrito por nosotros mientras ajustábamos el sistema.
# LoCoMo es ajeno: 10 conversaciones largas (19-32 sesiones cada una) entre dos personas, con 1.986 preguntas
# de referencia en 5 categorías:
#   1 multi-salto · 2 temporal · 3 inferencia (sentido común) · 4 un salto · 5 adversarial
# La categoría 5 atribuye a una persona algo que le ocurrió a OTRA (o que no consta): es exactamente el fallo
# conocido de la «hija pequeña», con 446 preguntas en vez de 31.
#
# Qué ejercita (por primera vez con datos ajenos):
#   * la INGESTA REAL: extract_facts() -> store() (colocación con LLM, dedupe, sustitución temporal),
#   * la memoria EPISÓDICA (cada turno se guarda literal, como en producción),
#   * la recuperación de producción (recall directo + search_episodes) y el prompt REAL del agente.
#
# Cómo se mide (mismo OBJETIVO que el resto del banco):
#   correctas = categorías 1-4 juzgadas contra la respuesta de referencia (juez LLM por mayoría)
#   inventa   = categoría 5: el agente afirma para la persona preguntada un dato que no consta
#   P95       = latencia de recuperación (recall + episódica)
# Diagnóstico: para cada fallo se decide si la evidencia (turnos D<s>:<t> que da LoCoMo) llegó al prompt,
# estaba entre los candidatos, existía como hecho en la memoria o se perdió en la ingesta.
#
# El dataset NO se redistribuye: se descarga en tiempo de ejecución del repositorio de los autores y se
# verifica su sha256. La memoria ingerida se guarda en experiment/locomo_cache/ (fuera de git) y la ingesta
# es REANUDABLE: si se interrumpe, al relanzar continúa por donde iba. Cambiar el código de ingesta, el
# modelo o el tamaño de bloque invalida la caché automáticamente.
#
# Variables:
#   HMEM_LOCOMO_CONV     conversaciones: índices "0,1" (defecto), ids "conv-26" o "all"
#   HMEM_LOCOMO_CHUNK    turnos por bloque de extracción (defecto 6)
#   HMEM_LOCOMO_MAXQ     máx. preguntas por conversación (0 = todas, defecto)
#   HMEM_LOCOMO_REF      preguntas para la referencia SIN memoria (defecto 30; 0 la desactiva)
#   HMEM_LOCOMO_REINGEST 1 = descarta la caché y reingiere
#   HMEM_LOCOMO_DATA     ruta a un locomo10.json local (sin descarga)
#   HMEM_LOCOMO_DIR      carpeta de caché (defecto experiment/locomo_cache)
#   HMEM_E2E_WORKERS     respuestas/juicios en paralelo (defecto 2)
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import config as C

ROOT = Path(__file__).resolve().parent.parent
VERSION = "locomo-v1"
DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
CATS = {1: "multi-salto", 2: "temporal", 3: "inferencia", 4: "un salto", 5: "adversarial"}
POS_CATS = (1, 2, 3, 4)
# Cifras publicadas (Mem0, arXiv 2504.19413, tabla 2: juez LLM, categorías 1-4, GPT-4o-mini).
# NO son comparables 1:1 (otro modelo, otro juez, otro prompt): sirven de orden de magnitud.
PUBLISHED = [("full-context", 72.90), ("Mem0ᵍ", 68.44), ("Mem0", 66.88), ("Zep", 65.99),
             ("RAG (mejor)", 60.97), ("LangMem", 58.10), ("OpenAI", 52.90), ("A-Mem", 48.38)]

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
_DT_RE = re.compile(r"\s*(\d{1,2}):(\d{2})\s*([ap]m)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\s*", re.I)
_EV_RE = re.compile(r"D:?(\d+):(\d+)")


# ==============================================================================
# 1. FORMATO DE LOCOMO (funciones puras, cubiertas por tests/test_locomo.py)
# ==============================================================================

def parse_dt(text: str) -> datetime:
    """'1:56 pm on 8 May, 2023' -> datetime (independiente del locale)."""
    m = _DT_RE.fullmatch(str(text))
    if not m:
        raise ValueError(f"fecha LoCoMo no reconocida: {text!r}")
    hh, mm, ap, dd, mon, yy = m.groups()
    h = int(hh) % 12 + (12 if ap.lower() == "pm" else 0)
    return datetime(int(yy), _MONTHS[mon.lower()], int(dd), h, int(mm))


def sessions(sample: dict) -> list[tuple[int, datetime, list[dict]]]:
    conv = sample["conversation"]
    out = []
    for k, v in conv.items():
        m = re.fullmatch(r"session_(\d+)", k)
        if m and isinstance(v, list) and v:
            out.append((int(m.group(1)), parse_dt(conv[f"{k}_date_time"]), v))
    return sorted(out, key=lambda s: s[0])


def turn_text(turn: dict) -> str:
    text = str(turn.get("text", "")).strip()
    cap = str(turn.get("blip_caption") or "").strip()
    return f"{text} [comparte una foto: {cap}]" if cap else text


def fecha(dt: datetime) -> str:
    return dt.strftime("%d/%m/%Y a las %H:%M")


def make_chunks(sample: dict, size: int = 6) -> list[dict]:
    """Bloques de `size` turnos consecutivos de una misma sesión, con cabecera de fecha y participantes."""
    size = max(1, int(size))
    a, b = sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]
    out = []
    for n, dt, turns in sessions(sample):
        for i in range(0, len(turns), size):
            part = turns[i:i + size]
            head = f"[Conversación entre {a} y {b} · sesión {n} · {fecha(dt)}]"
            lines = [f"{t['speaker']}: {turn_text(t)}" for t in part]
            out.append({"session": n, "dt": dt, "offset": i, "turns": part,
                        "dia_ids": [str(t["dia_id"]) for t in part], "text": head + "\n" + "\n".join(lines)})
    return out


def norm_evidence(ev: Any) -> set[str]:
    """Normaliza las evidencias ('D8:6; D9:17', 'D:11:26', 'D9:1 D4:4') a {'D8:6', ...}."""
    items = ev if isinstance(ev, list) else [ev]
    return {f"D{s}:{t}" for it in items for s, t in _EV_RE.findall(str(it))}


def _sha_file(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return "-"


def cache_key(sample_id: str, size: int, models: tuple[str, ...] = ()) -> str:
    """Cambia si cambia el código de ingesta, los modelos, el tamaño de bloque o la versión del adaptador."""
    h = hashlib.sha256()
    for part in (VERSION, sample_id, str(size), *models):
        h.update(str(part).encode("utf-8") + b"\0")
    for rel in ("src/ltm/ingestion.py", "src/ltm/storage.py", "src/ltm/common.py", "src/episodica.py"):
        h.update(_sha_file(ROOT / rel).encode())
    return h.hexdigest()[:10]


def select_samples(data: list[dict], spec: str) -> list[dict]:
    spec = (spec or "").strip().lower()
    if spec in ("all", "todas", "*"):
        return list(data)
    out = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok.isdigit() and int(tok) < len(data):
            out.append(data[int(tok)])
        else:
            out += [s for s in data if str(s.get("sample_id", "")).lower() == tok]
    return out


def subsample(qa: list[dict], maxq: int, seed: int = 10) -> list[dict]:
    """Muestra ESTRATIFICADA por categoría (LoCoMo ordena las preguntas por categoría: cortar las primeras N
    dejaría fuera las adversariales). Determinista y en el orden original."""
    if not maxq or maxq >= len(qa):
        return list(qa)
    by_cat: dict[int, list[int]] = {}
    for i, q in enumerate(qa):
        by_cat.setdefault(int(q.get("category", 0)), []).append(i)
    quota = {c: max(1, round(maxq * len(ix) / len(qa))) for c, ix in by_cat.items()}
    while sum(quota.values()) > maxq and max(quota.values()) > 1:
        quota[max(quota, key=quota.get)] -= 1
    rnd = random.Random(seed)
    keep = {i for c, ix in by_cat.items() for i in rnd.sample(ix, min(quota[c], len(ix)))}
    return [q for i, q in enumerate(qa) if i in keep]


# ---- [PARCHE11] diagnóstico de identidad del sujeto y de top-K
def subject_of(question: str, a: str, b: str) -> str | None:
    """Persona por la que se pregunta: el interlocutor que aparece ANTES en la pregunta (admite 'Mel' por 'Melanie')."""
    best: tuple[int, str] | None = None
    for name in (a, b):
        for m in re.finditer(r"[A-Za-zÀ-ÿ]+", question):
            w = m.group(0).lower()
            if len(w) >= 3 and name.lower().startswith(w) and (best is None or m.start() < best[0]):
                best = (m.start(), name)
                break
    return best[1] if best else None


def fact_lines(ltm_answer: str) -> list[str]:
    return [ln[1:].strip() for ln in str(ltm_answer or "").splitlines() if ln.strip().startswith("•")]


def _mentions(text: str, name: str) -> bool:
    return bool(name) and any(len(w) >= 3 and name.lower().startswith(w.lower())
                              for w in re.findall(r"[A-Za-zÀ-ÿ]+", text))


def phrasing(lines: list[str], asked: str | None, other: str | None) -> dict[str, int]:
    """Clasifica cada hecho inyectado: nombra al sujeto preguntado, nombra SOLO a la otra persona,
    habla de 'el usuario' sin nombre, o no identifica a nadie."""
    out = {"sujeto": 0, "otra": 0, "usuario": 0, "sin sujeto": 0}
    for ln in lines:
        body = re.sub(r"\(Fuente:.*?\)\s*$", "", re.sub(r"^\[[^\]]*\]\s*", "", ln))   # sin ruta ni fuente
        if asked and _mentions(body, asked):
            out["sujeto"] += 1
        elif other and _mentions(body, other):
            out["otra"] += 1
        elif re.search(r"\busuari[oa]s?\b", body, re.I):
            out["usuario"] += 1
        else:
            out["sin sujeto"] += 1
    return out


def evidence_rank(candidates: list[Any], cell_dia: dict[str, list[str]], ev: set[str]) -> int | None:
    """Posición (1-based) del primer candidato cuyo bloque contiene evidencia."""
    for i, cid in enumerate(candidates, 1):
        if ev & set(cell_dia.get(str(cid), [])):
            return i
    return None


def coverage(ev: set[str], cells: list[Any], epis: list[dict], cell_dia: dict[str, list[str]],
             epi_dia: dict[str, str]) -> float:
    """Fracción de los turnos de evidencia presentes en el contexto (a nivel de bloque para los hechos)."""
    if not ev:
        return 0.0
    got = {d for c in cells for d in cell_dia.get(str(c), [])} | {epi_dia.get(str(e.get("id"))) for e in epis}
    return len(ev & got) / len(ev)


# ---- [PARCHE12] comparación pareada con una ejecución anterior y auditoría automática de invenciones
def sign_p(b: int, c: int) -> float:
    """Test de signos exacto bilateral (McNemar exacto) sobre los pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare_runs(base: list[dict], new: list[dict]) -> dict[str, dict[str, Any]]:
    """Compara pregunta a pregunta (sid, pregunta). Filas: {sid, q, cat, veredicto}.
    pos: 'bien' = correcta · neg: 'bien' = NO inventa (sin veredicto cuenta como inventa)."""
    good = lambda r: (r.get("veredicto") is True) if int(r["cat"]) in POS_CATS else (r.get("veredicto") is False)

    def keyed(rows: list[dict]) -> dict[tuple, dict]:
        # LoCoMo repite algunas preguntas en la misma conversación: se empareja por n.º de aparición
        seen: dict[tuple, int] = {}
        out_k: dict[tuple, dict] = {}
        for r in rows:
            if r.get("ref"):
                continue
            k = (r["sid"], r["q"], int(r["cat"]))
            seen[k] = seen.get(k, 0) + 1
            out_k[k + (seen[k],)] = r
        return out_k

    old, cur = keyed(base), keyed(new)
    out: dict[str, dict[str, Any]] = {}
    for name, cats in (("pos", POS_CATS), ("neg", (5,))):
        pairs = [(old[k], r) for k, r in cur.items() if int(r["cat"]) in cats and k in old]
        b = sum(1 for o, n in pairs if good(o) and not good(n))      # empeora
        c = sum(1 for o, n in pairs if not good(o) and good(n))      # mejora
        out[name] = {"n": len(pairs), "antes": sum(good(o) for o, _ in pairs), "ahora": sum(good(n) for _, n in pairs),
                     "empeora": b, "mejora": c, "p": sign_p(b, c)}
    return out


def trap_source(trap: str, facts: list[str], episodes: list[str]) -> str:
    """¿De dónde sale el dato trampa que repite el agente: episódica, hechos o ninguno claro?"""
    words = {w for w in re.findall(r"[a-záéíóúñ]{4,}", str(trap).lower())}
    if not words:
        return "?"
    ep, fa = " ".join(episodes).lower(), " ".join(facts).lower()
    e = sum(w in ep for w in words) / len(words)
    f = sum(w in fa for w in words) / len(words)
    return "episódica" if e >= 0.5 and e >= f else ("hechos" if f >= 0.5 else "ninguno claro")


def evidence_bucket(in_ctx: bool, in_cand: bool, has_cell: bool) -> str:
    """Dónde se perdió la evidencia de una respuesta fallida."""
    if in_ctx:
        return "respuesta"       # estaba en el prompt: falló el LLM al usarla (o el juez)
    if in_cand:
        return "selección"       # estaba entre los candidatos pero no se inyectó (compuerta / top-6)
    if has_cell:
        return "recuperación"    # existe un hecho de ese bloque, pero la búsqueda no lo trajo
    return "ingesta"             # ningún hecho del bloque de la evidencia (y la episódica tampoco la trajo)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


_ES = re.compile(r"\b(el|la|los|las|del|que|una|con|por|para|su|sus|fue|está|también)\b", re.I)
_EN = re.compile(r"\b(the|and|her|his|she|he|was|with|for|of|that|their|is)\b", re.I)


def language(text: str) -> str:
    es, en = len(_ES.findall(text)), len(_EN.findall(text))
    return "es" if es > en else "en" if en > es else "?"


# Muestra mínima en el formato de LoCoMo para el modo emulado y los tests (sin red, sin datos ajenos).
MINI_SAMPLE: dict[str, Any] = {
    "sample_id": "mini-1",
    "conversation": {
        "speaker_a": "Lucas", "speaker_b": "Nora",
        "session_1_date_time": "10:00 am on 3 March, 2024",
        "session_1": [
            {"speaker": "Lucas", "dia_id": "D1:1", "text": "Hi Nora! I adopted a ferret named Pimienta last week."},
            {"speaker": "Nora", "dia_id": "D1:2", "text": "That is lovely! I started pottery classes in Burgos."},
            {"speaker": "Lucas", "dia_id": "D1:3", "text": "Nice. My sister Elvira moved to Lisbon."},
            {"speaker": "Nora", "dia_id": "D1:4", "text": "I have been reading a novel about lighthouses.",
             "blip_caption": "a photo of a lighthouse at sunset"},
        ],
        "session_2_date_time": "6:30 pm on 20 April, 2024",
        "session_2": [
            {"speaker": "Nora", "dia_id": "D2:1", "text": "I won the regional chess tournament yesterday!"},
            {"speaker": "Lucas", "dia_id": "D2:2", "text": "Congrats! Pimienta learned to open the fridge."},
            {"speaker": "Nora", "dia_id": "D2:3", "text": "Next month I am travelling to Kyoto."},
        ],
    },
    "qa": [
        {"question": "What is the name of Lucas's ferret?", "answer": "Pimienta", "evidence": ["D1:1"], "category": 4},
        {"question": "Where did Nora start pottery classes?", "answer": "Burgos", "evidence": ["D1:2"], "category": 4},
        {"question": "When did Nora win the chess tournament?", "answer": "19 April 2024", "evidence": ["D2:1"],
         "category": 2},
        {"question": "What pets and travels are mentioned?", "answer": "a ferret; a trip to Kyoto",
         "evidence": ["D1:1", "D2:3"], "category": 1},
        {"question": "Where did Lucas's sister Elvira move?", "answer": "Lisbon", "evidence": ["D1:3"], "category": 4},
        {"question": "What did Lucas win at the chess tournament?", "adversarial_answer": "the regional title",
         "evidence": ["D2:1"], "category": 5},
        {"question": "What is the name of Nora's ferret?", "adversarial_answer": "Pimienta",
         "evidence": ["D1:1"], "category": 5},
    ],
}


# ==============================================================================
# 2. DATOS
# ==============================================================================

def cache_root() -> Path:
    return Path(os.getenv("HMEM_LOCOMO_DIR") or (ROOT / "experiment" / "locomo_cache"))


def load_dataset(live: bool) -> tuple[list[dict] | None, str]:
    """Devuelve (datos, descripción). En modo emulado usa MINI_SAMPLE salvo que se indique HMEM_LOCOMO_DATA."""
    local = os.getenv("HMEM_LOCOMO_DATA", "").strip()
    if not live and not local:
        return [MINI_SAMPLE], "muestra mínima embebida (modo emulado)"
    path = Path(local).expanduser() if local else cache_root() / "locomo10.json"
    if not path.exists():
        if local:
            return None, f"no existe {path}"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.part")
        try:
            print(f"   … descargando LoCoMo de {DATA_URL}", flush=True)
            with urllib.request.urlopen(DATA_URL, timeout=90) as resp, tmp.open("wb") as fo:
                shutil.copyfileobj(resp, fo)
            tmp.replace(path)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            return None, (f"no se pudo descargar ({type(exc).__name__}: {exc}). Descárgalo a mano en {path} "
                          f"o indica HMEM_LOCOMO_DATA")
    sha = _sha_file(path)
    note = "sha256 verificado" if sha == DATA_SHA256 else f"⚠ sha256 DISTINTO al esperado ({sha[:12]}…): el dataset cambió"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{path} ilegible: {exc}"
    return data, f"{path} ({note})"


# ==============================================================================
# 3. INGESTA REANUDABLE POR EL CAMINO DE PRODUCCIÓN
# ==============================================================================

def new_state(key: str, n_chunks: int) -> dict[str, Any]:
    return {"version": VERSION, "key": key, "done": 0, "n_chunks": n_chunks, "cell_dia": {}, "epi_dia": {},
            "stats": {"turns": 0, "facts": 0, "dup": 0, "fail": 0, "calls": 0, "secs": 0.0,
                      "superseded": [], "complete": False}}


def _episode_id(tree: Any, role: str, text: str, ts: float) -> int | None:
    eid = tree.log_episode(role, text, ts)
    if eid is not None:
        return int(eid)
    if getattr(tree, "episodes", None) is None:
        return None
    uid = hashlib.sha1(f"{ts:.3f}|{role}|{text}".encode("utf-8")).hexdigest()   # mismo uid que episodica.py
    with tree.lock:
        r = tree.conn.execute("SELECT id FROM episodes WHERE uid = ?", (uid,)).fetchone()
    return int(r[0]) if r else None


def ingest(tree: Any, sample: dict, size: int, state: dict[str, Any],
           save: Callable[[dict[str, Any]], None] | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Ingiere la conversación bloque a bloque: episodios literales + extract_facts() + store()."""
    chunks = make_chunks(sample, size)
    a, b = sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]
    st = state["stats"]
    state["n_chunks"] = len(chunks)
    t_start, done0 = time.time(), int(state["done"])
    for ci in range(done0, len(chunks)):
        ch = chunks[ci]
        t0, calls0 = time.time(), getattr(tree.llm, "calls", 0)
        base_ts = ch["dt"].timestamp()
        for j, turn in enumerate(ch["turns"]):   # memoria episódica: cada turno, literal, con su fecha
            eid = _episode_id(tree, "user", f"{turn['speaker']}: {turn_text(turn)}", base_ts + ch["offset"] + j)
            if eid is not None:
                state["epi_dia"][str(eid)] = str(turn["dia_id"])
        facts = None
        for _attempt in range(2):
            _src, facts = tree.extract_facts(ch["text"], first=False, origin="conversación")
            if facts is not None:
                break
        if facts is None:
            st["fail"] += 1
            facts = []
        ts_str = ch["dt"].strftime("%Y-%m-%d %H:%M:%S")
        meta = {"fuente": f"Conversación entre {a} y {b} el {fecha(ch['dt'])}",
                "ingest_timestamp": ts_str, "valid_from": ts_str}
        for f in facts:   # igual que IngestionMixin._ingest_loop
            imp = max(9.0, f["importancia"]) if f["tipo"] == "instruccion" else f["importancia"]
            try:
                res = tree.store(f["texto"], imp, f["tipo"], meta=dict(meta))
            except Exception as exc:
                st["fail"] += 1
                log(f"   ! store() falló: {type(exc).__name__}: {exc}")
                continue
            cid = str(res.get("id") or "")
            if not cid:
                continue
            st["facts"] += 1
            st["dup"] += bool(res.get("duplicate"))
            dias = state["cell_dia"].setdefault(cid, [])
            dias += [d for d in ch["dia_ids"] if d not in dias]
            if res.get("superseded"):
                old = res["superseded"]
                with tree.lock:
                    r = tree.conn.execute("SELECT text FROM cells WHERE id = ?", (old,)).fetchone()
                st["superseded"].append({"old": r[0] if r else old, "new": f["texto"], "dia": ch["dia_ids"]})
        st["turns"] += len(ch["turns"])
        st["calls"] += getattr(tree.llm, "calls", 0) - calls0
        st["secs"] += time.time() - t0
        state["done"] = ci + 1
        if save:
            save(state)
        n_new = ci + 1 - done0
        if n_new % 10 == 0 or ci + 1 == len(chunks):
            rate = (time.time() - t_start) / n_new
            log(f"   · bloque {ci + 1}/{len(chunks)} · {st['facts']} hechos · "
                f"{st['calls']} llamadas LLM · quedan ≈{rate * (len(chunks) - ci - 1) / 60:.0f} min")
    st["complete"] = state["done"] >= len(chunks)
    if save:
        save(state)
    return state


# ==============================================================================
# 4. RECUPERACIÓN, PROMPT Y JUECES
# ==============================================================================

def retrieve(tree: Any, question: str, state: dict[str, Any], evidence: set[str]) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = tree.recall(question, reinforce=False, synthesize=False)
    t1 = time.perf_counter()
    epi = tree.search_episodes(question, int(getattr(C, "EPI_LOOKUP_K", 6))) or []
    t2 = time.perf_counter()
    cell_dia, epi_dia = state["cell_dia"], state["epi_dia"]
    hit = lambda ids: any(evidence & set(cell_dia.get(str(i), [])) for i in ids)
    hit_epi = lambda eps: any(epi_dia.get(str(e.get("id"))) in evidence for e in eps)
    injected = list(r.get("cells") or []) if r.get("found") else []
    top_sim = None   # [PARCHE11] similitud coseno del mejor hecho (fuera del tiempo medido)
    emb = getattr(tree, "emb", None)
    if emb is not None:
        try:
            qv = emb.embed_query(question)
            hits = emb.store.search("cell", qv, 1, -1.0, emb.model) if qv is not None else []
            top_sim = round(float(hits[0][1]), 4) if hits else None
        except Exception:
            top_sim = None
    return {
        "top_sim": top_sim,
        "ev_rank": evidence_rank(r.get("candidates") or [], cell_dia, evidence),
        "coverage": coverage(evidence, injected, epi[:5], cell_dia, epi_dia),
        "lines": fact_lines(r.get("answer", "")) if r.get("found") else [],
        "r": r, "epi": epi, "ms": (t2 - t0) * 1000, "ms_recall": (t1 - t0) * 1000,
        "state": "rechaza" if not r.get("found") else ("tentativa" if r.get("tentative") else "firme"),
        "in_ctx": bool(evidence) and (hit(injected) or hit_epi(epi[:5])),
        "in_cand": bool(evidence) and (hit(r.get("candidates") or []) or hit_epi(epi)),
        "has_cell": bool(evidence) and any(evidence & set(v) for v in cell_dia.values()),
    }


def system_prompt(ret: dict[str, Any]) -> str:
    from agente import Agent
    r = ret["r"]
    found, tent, ans = bool(r.get("found")), bool(r.get("tentative")), str(r.get("answer") or "")
    ltm = {"found": found, "tentative": tent, "sources": list(r.get("sources") or []),
           "answer": "" if tent else (ans if found else ""), "tentative_answer": ans if (found and tent) else ""}
    return Agent._build_system(None, {}, ltm, ret["epi"])


LOCOMO_POS_JUDGE = (
    "Eres un evaluador. Se hizo una pregunta a un asistente con memoria sobre una conversación pasada y existe una "
    "RESPUESTA DE REFERENCIA. Decide si la respuesta del asistente es correcta: contiene lo esencial de la referencia "
    "(aunque use otras palabras u otro idioma, o añada detalles correctos) y no la contradice. Una fecha es correcta si "
    "designa el mismo día, semana, mes o año que la referencia, aunque se exprese de otra forma. Si la referencia es una "
    "lista, basta con que acierte la mayoría de sus elementos. Si dice que no lo sabe, o da como respuesta principal "
    "otro dato, es incorrecta.\n"
    'Devuelve SOLO JSON: {"correcta": true} o {"correcta": false}.')
LOCOMO_NEG_JUDGE = (
    "Eres un evaluador. Se hizo a un asistente con memoria una pregunta TRAMPA sobre una conversación pasada: atribuye "
    "a una persona algo que, según la conversación, no consta de ella (a menudo le ocurrió a otra persona). Decide si "
    "la respuesta INVENTA: afirma como cierto para la persona preguntada un dato concreto (la respuesta trampa indicada "
    "u otro valor). NO es inventar: decir que no consta o que no lo sabe, ni corregir la premisa (por ejemplo, señalar "
    "que fue otra persona).\n"
    'Devuelve SOLO JSON: {"inventa": true} o {"inventa": false}.')


def _vote(chat: Any, system: str, user: str, key: str) -> bool | None:
    def one() -> bool | None:
        try:
            obj = chat.ask_json(system, user)
            return bool(obj.get(key)) if isinstance(obj, dict) and key in obj else None
        except Exception:
            return None
    v1, v2 = one(), one()
    if v1 is not None and v1 == v2:
        return v1
    votes = [v for v in (v1, v2, one()) if v is not None]
    return (2 * sum(votes) > len(votes)) if votes else None


# ==============================================================================
# 5. SUITE
# ==============================================================================

def run(ctx: Any, open_tree: Callable[[Path], Any], sync_all: Callable[[Any], None],
        north_star: dict[str, float]) -> None:
    R, S = ctx.report, "locomo"
    data, info = load_dataset(ctx.live)
    if data is None:
        R.suite_skipped[S] = info
        return
    spec = os.getenv("HMEM_LOCOMO_CONV", "0,1") if ctx.live or os.getenv("HMEM_LOCOMO_DATA") else "all"
    samples = select_samples(data, spec)
    if not samples:
        R.suite_skipped[S] = f"HMEM_LOCOMO_CONV={spec!r} no selecciona ninguna conversación"
        return
    size = max(1, int(os.getenv("HMEM_LOCOMO_CHUNK", "6")))
    maxq = max(0, int(os.getenv("HMEM_LOCOMO_MAXQ", "0")))
    n_ref = max(0, int(os.getenv("HMEM_LOCOMO_REF", "30")))
    workers = max(1, int(os.getenv("HMEM_E2E_WORKERS", "2")))
    reingest = os.getenv("HMEM_LOCOMO_REINGEST", "").strip().lower() in {"1", "true", "yes", "si", "sí"}
    models = (str(getattr(C, "REMOTE_MODEL_ALIAS", "")), str(getattr(C, "EMB_MODEL_ALIAS", "")),
              "mock" if ctx.mock else "live")
    base = cache_root() if ctx.live else Path(ctx.tmp) / "locomo"
    print(f"   datos: {info}")
    print(f"   conversaciones: {', '.join(s['sample_id'] for s in samples)} · bloque {size} turnos · "
          f"caché: {base}")

    items: list[dict[str, Any]] = []
    ingest_stats: list[dict[str, Any]] = []
    all_facts: list[str] = []
    for sample in samples:
        sid = str(sample["sample_id"])
        key = cache_key(sid, size, models)
        d = base / f"{sid}-{key}"
        state_file = d / "estado.json"
        if reingest and d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state.get("key") != key:
                raise ValueError("clave distinta")
        except Exception:
            for f in d.glob("memoria.db*"):
                f.unlink(missing_ok=True)
            state = new_state(key, 0)

        def save(st: dict[str, Any], _f: Path = state_file) -> None:
            tmp = _f.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, _f)

        tree = open_tree(d)
        n_turns = sum(len(t) for _n, _dt, t in sessions(sample))
        n_chunks = len(make_chunks(sample, size))
        if not state["stats"].get("complete"):
            verb = "reanudando" if state["done"] else "ingiriendo"
            print(f"   … {sid}: {verb} {n_turns} turnos en {n_chunks} bloques por extract_facts() → store() "
                  f"(bloque {state['done'] + 1}/{n_chunks})", flush=True)
            ingest(tree, sample, size, state, save=save)
        else:
            print(f"   · {sid}: memoria en caché ({state['stats']['facts']} hechos)")
        sync_all(tree)
        st = dict(state["stats"])
        with tree.lock:
            st["alive"] = int(tree.conn.execute("SELECT COUNT(*) FROM cells WHERE archived = 0").fetchone()[0])
            st["nodes"] = int(tree.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
            facts_txt = [r[0] for r in tree.conn.execute("SELECT text FROM cells").fetchall()]
        st["sid"], st["n_turns"] = sid, n_turns
        ingest_stats.append(st)
        all_facts += facts_txt

        # ---- recuperación (secuencial: la latencia se mide sin contención)
        qa = subsample(list(sample["qa"]), maxq)
        spk_of = {str(t["dia_id"]): str(t["speaker"]) for _n, _dt, ts in sessions(sample) for t in ts}  # [PARCHE12]
        print(f"   … {sid}: recuperando {len(qa)} preguntas", flush=True)
        for q in qa:
            cat = int(q.get("category", 0))
            ev = norm_evidence(q.get("evidence", []))
            ret = retrieve(tree, str(q["question"]), state, ev)
            spk_a, spk_b = sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]
            subj = subject_of(str(q["question"]), spk_a, spk_b)
            other = (spk_b if subj == spk_a else spk_a) if subj else None
            ev_spk = sorted({spk_of[d] for d in ev if d in spk_of})   # [PARCHE12] quién pronunció la evidencia
            items.append({"subj": subj, "other": other, "ev_spk": ev_spk,
                          "ph": phrasing(ret["lines"], subj, other),
                          "sid": sid, "q": str(q["question"]), "cat": cat, "ev": ev,
                          "gold": str(q.get("answer", "")), "trap": str(q.get("adversarial_answer", "")),
                          "ret": ret, "system": system_prompt(ret)})

    pos = [it for it in items if it["cat"] in POS_CATS]
    neg = [it for it in items if it["cat"] == 5]

    # ---- ingesta
    tot = {k: sum(float(s.get(k, 0)) for s in ingest_stats) for k in ("turns", "facts", "dup", "fail", "calls", "secs")}
    n_sup = sum(len(s.get("superseded", [])) for s in ingest_stats)
    usuario = sum(1 for t in all_facts if re.search(r"\busuario\b", t, re.I)) / max(1, len(all_facts))
    langs = {k: sum(1 for t in all_facts if language(t) == k) / max(1, len(all_facts)) for k in ("es", "en")}
    print(f"\n   INGESTA (camino de producción · {len(ingest_stats)} conversación(es))")
    for s in ingest_stats:
        print(f"   {s['sid']:<9} {s['n_turns']:>4} turnos → {int(s['facts']):>5} hechos guardados "
              f"({int(s['dup'])} duplicados, {len(s.get('superseded', []))} sustituciones) · {s['alive']} vivos en "
              f"{s['nodes']} nodos · {int(s['fail'])} fallos")
    if tot["secs"] > 0:
        print(f"   ritmo: {tot['turns'] / tot['secs'] * 60:.1f} turnos/min · {tot['facts'] / tot['secs'] * 60:.1f} hechos/min "
              f"· {tot['calls'] / max(1, tot['facts']):.1f} llamadas LLM por hecho · {tot['secs'] / 60:.0f} min de ingesta "
              f"acumulada")
    print(f"   hechos que hablan de 'el usuario' (no hay usuario en LoCoMo): {pct(usuario)} · idioma de los hechos: "
          f"es {pct(langs['es'])} · en {pct(langs['en'])}")
    sups = [x for s in ingest_stats for x in s.get("superseded", [])]
    if sups:
        print("   · sustituciones temporales (auditar: ¿mismo sujeto y atributo?):")
        for x in sups[:8]:
            print(f"     «{x['old'][:60]}» ⇒ «{x['new'][:60]}»")
    R.add(S, "Ingesta: fallos de extracción/almacenamiento", tot["fail"] == 0, f"{int(tot['fail'])}", "0",
          "medio" if ctx.live else "info", "ltm/ingestion.py · extract_facts / ltm/storage.py · store")
    if tot["secs"] > 0:
        R.add(S, "Ingesta: hechos por minuto", True, f"{tot['facts'] / tot['secs'] * 60:.1f}", "(observación)", "info",
              "ltm/storage.py · store", f"{tot['calls'] / max(1, tot['facts']):.1f} llamadas LLM por hecho")
    R.add(S, "Ingesta: sustituciones temporales", True, str(n_sup), "(auditar)", "info",
          "ltm/storage.py · _detect_temporal_supersession",
          evidence=[f"«{x['old'][:50]}» ⇒ «{x['new'][:50]}»" for x in sups[:4]])
    R.add(S, "Hechos redactados como 'el usuario'", True, pct(usuario), "(observación)", "info",
          "ltm/ingestion.py · extract_facts (regla 1)",
          "LoCoMo no tiene 'usuario': si es alto, la extracción asume un único interlocutor.")
    R.add(S, "Idioma de los hechos (es / en)", True, f"{pct(langs['es'])} / {pct(langs['en'])}", "(observación)",
          "info", "ltm/ingestion.py · extract_facts")

    # ---- recuperación
    lat = [it["ret"]["ms"] for it in items]
    ev_pos = [it for it in pos if it["ev"]]
    ctx_rate = sum(it["ret"]["in_ctx"] for it in ev_pos) / max(1, len(ev_pos))
    cand_rate = sum(it["ret"]["in_cand"] for it in ev_pos) / max(1, len(ev_pos))
    cnt = lambda xs, k: sum(1 for it in xs if it["ret"]["state"] == k)
    print(f"\n   RECUPERACIÓN ({len(items)} preguntas: {len(pos)} con respuesta · {len(neg)} adversariales)")
    print(f"   evidencia en el prompt {pct(ctx_rate)} · entre los candidatos {pct(cand_rate)} "
          f"(a nivel de bloque de {size} turnos) · P95 {percentile(lat, 95):.0f} ms "
          f"(solo recall {percentile([it['ret']['ms_recall'] for it in items], 95):.0f} ms)")
    print(f"   compuerta · con respuesta: firme {cnt(pos, 'firme')} / tentativa {cnt(pos, 'tentativa')} / rechaza "
          f"{cnt(pos, 'rechaza')} · adversariales: firme {cnt(neg, 'firme')} / tentativa {cnt(neg, 'tentativa')} / "
          f"rechaza {cnt(neg, 'rechaza')}")
    print(f"   {'categoría':<14}{'n':>5}{'evid. prompt':>14}{'evid. cand.':>13}")
    for c in POS_CATS:
        sel = [it for it in ev_pos if it["cat"] == c]
        if sel:
            print(f"   {CATS[c]:<14}{len(sel):>5}{pct(sum(it['ret']['in_ctx'] for it in sel) / len(sel)):>14}"
                  f"{pct(sum(it['ret']['in_cand'] for it in sel) / len(sel)):>13}")
    R.add(S, "Evidencia en el prompt (categorías 1-4)", True, pct(ctx_rate), "(observación)", "info",
          "ltm/retrieval.py · recall / episodica.py", f"entre candidatos: {pct(cand_rate)}")
    R.add(S, "LoCoMo · P95 recuperación (recall + episódica)", percentile(lat, 95) <= north_star["p95_ms"],
          f"{percentile(lat, 95):.0f} ms", f"≤ {north_star['p95_ms']:.0f} ms", "medio" if ctx.live else "info",
          "vectores.py / ltm/retrieval.py / episodica.py")
    # ---- [PARCHE11] top-K, cobertura, similitud de la compuerta e identidad de los hechos (sin LLM)
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    rk = [it["ret"]["ev_rank"] for it in ev_pos]
    k_in, k_mid, k_out = (sum(1 for x in rk if x and x <= 6), sum(1 for x in rk if x and x > 6),
                          sum(1 for x in rk if not x))
    print(f"\n   TOP-K: el primer hecho con evidencia está en el puesto 1-6 en {k_in} · 7-18 en {k_mid} "
          f"(candidato pero no se inyecta) · fuera de los 18 en {k_out}  (n={len(rk)})")
    print("   cobertura media de los turnos de evidencia en el contexto: " + " · ".join(
        f"{CATS[c]} {pct(avg([it['ret']['coverage'] for it in ev_pos if it['cat'] == c]))}"
        for c in POS_CATS if any(it["cat"] == c for it in ev_pos)))
    sims = lambda xs: sorted(it["ret"]["top_sim"] for it in xs if it["ret"]["top_sim"] is not None)
    sp, sn = sims(pos), sims(neg)
    if sp and sn:
        print(f"   similitud top-1 · con respuesta P10/P50/P90 {percentile(sp, 10):.2f}/{percentile(sp, 50):.2f}/"
              f"{percentile(sp, 90):.2f} · adversariales {percentile(sn, 10):.2f}/{percentile(sn, 50):.2f}/"
              f"{percentile(sn, 90):.2f}  (si se solapan, ningún umbral de la compuerta separa unas de otras)")
        R.add(S, "Similitud top-1 P50 (con respuesta / adversariales)", True,
              f"{percentile(sp, 50):.2f} / {percentile(sn, 50):.2f}", "(diagnóstico)", "info",
              "ltm/retrieval.py · _classify_relevance")
    tot_ph = {k: sum(it["ph"][k] for it in items) for k in ("sujeto", "otra", "usuario", "sin sujeto")}
    n_ph = max(1, sum(tot_ph.values()))
    print("   IDENTIDAD de los hechos inyectados: " + " · ".join(f"{k} {pct(v / n_ph)}" for k, v in tot_ph.items())
          + f"  (n={sum(tot_ph.values())} hechos; 'usuario' = no se sabe de quién habla)")
    R.add(S, "Hechos inyectados sin identidad ('el usuario')", True, pct(tot_ph["usuario"] / n_ph), "(diagnóstico)",
          "info", "ltm/ingestion.py · extract_facts (regla 1)")
    R.add(S, "Evidencia en puestos 7-18 (candidata, no inyectada)", True, f"{k_mid}/{len(rk)}", "(diagnóstico)",
          "info", "ltm/retrieval.py · modo directo (top-6)")

    lost_ctx = [it for it in items if it["ret"]["state"] != "rechaza" and "Memoria a Largo Plazo" not in it["system"]
                and "posible relevancia" not in it["system"]]
    R.add(S, "Lo recuperado llega al prompt real del agente", bool(items) and not lost_ctx,
          f"{len(items) - len(lost_ctx)}/{len(items)}", "todas", "alto", "agente.py · _build_system",
          evidence=[f"«{it['q'][:60]}»" for it in lost_ctx[:4]])

    if not ctx.live:
        print("   ⚠ modo emulado: valida la fontanería (ingesta, mapas de evidencia, prompt); "
              "las respuestas y el juez requieren --live")
        return

    # ---- respuestas reales del agente + juez
    from llm import LLM
    from agente import BASE_PROMPT
    chat = LLM(base_url=C.LLM_STM_URL, model=C.REMOTE_MODEL_ALIAS)
    rnd = random.Random(10)
    ref_items = rnd.sample(pos, min(n_ref, len(pos))) if n_ref else []
    jobs = [(it, it["system"]) for it in items] + [({**it, "ref": True}, BASE_PROMPT.format(now=time.strftime(
        "%d/%m/%Y %H:%M"))) for it in ref_items]

    def answer(job: tuple[dict, str]) -> str:
        it, system = job
        try:
            return chat.chat([{"role": "system", "content": system}, {"role": "user", "content": it["q"]}],
                             max_tokens=350, temperature=0.2)
        except Exception as exc:
            return f"[error LLM: {type(exc).__name__}]"

    def judge(it: dict) -> bool | None:
        if it["cat"] == 5:
            return _vote(chat, LOCOMO_NEG_JUDGE, f"Pregunta: {it['q']}\nRespuesta trampa: {it['trap']}\n"
                                                 f"Respuesta del asistente: {it['a']}", "inventa")
        return _vote(chat, LOCOMO_POS_JUDGE, f"Pregunta: {it['q']}\nRespuesta de referencia: {it['gold']}\n"
                                             f"Respuesta del asistente: {it['a']}", "correcta")

    t0 = time.time()
    print(f"\n   … {len(jobs)} respuestas del agente real + juez por mayoría ({workers} en paralelo)", flush=True)
    done_items: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for k, ((it, _s), a) in enumerate(zip(jobs, ex.map(answer, jobs)), 1):
            it["a"] = a
            done_items.append(it)
            if k % 50 == 0:
                print(f"     {k}/{len(jobs)} respuestas", flush=True)
        for it, v in zip(done_items, ex.map(judge, done_items)):
            it["v"] = v
    print(f"   listo ({(time.time() - t0) / 60:.0f} min)")
    main_items = [it for it in done_items if not it.get("ref")]
    ref_done = [it for it in done_items if it.get("ref")]
    pos = [it for it in main_items if it["cat"] in POS_CATS]
    neg = [it for it in main_items if it["cat"] == 5]

    ok = lambda it: it.get("v") is True
    inv = lambda it: it.get("v") is not False      # sin veredicto = inventa (criterio conservador)
    c_ok = sum(map(ok, pos))
    n_inv = sum(map(inv, neg))
    c_lo, c_hi = wilson(c_ok, len(pos))
    i_lo, i_hi = wilson(n_inv, len(neg))
    print(f"\n   🎯 OBJETIVO EN LOCOMO  ({len(pos)} preguntas con respuesta · {len(neg)} adversariales)")
    print(f"   correctas {c_ok}/{len(pos)} = {pct(c_ok / max(1, len(pos)))} [IC95 {pct(c_lo)}–{pct(c_hi)}] "
          f"(meta ≥ {pct(north_star['correctas'])}) · inventa {n_inv}/{len(neg)} = {pct(n_inv / max(1, len(neg)))} "
          f"[IC95 {pct(i_lo)}–{pct(i_hi)}] (meta ≤ {pct(north_star['inventa'])}) · P95 {percentile(lat, 95):.0f} ms")
    print(f"   {'categoría':<14}{'n':>5}{'correctas':>11}")
    for c in POS_CATS:
        sel = [it for it in pos if it["cat"] == c]
        if sel:
            print(f"   {CATS[c]:<14}{len(sel):>5}{pct(sum(map(ok, sel)) / len(sel)):>11}")
            R.add(S, f"LoCoMo · correctas · {CATS[c]}", True, f"{pct(sum(map(ok, sel)) / len(sel))} (n={len(sel)})",
                  "(observación)", "info", "")
    if ref_done:
        r_ok = sum(map(ok, ref_done))
        print(f"   referencia SIN memoria: {r_ok}/{len(ref_done)} = {pct(r_ok / len(ref_done))} "
              f"(cuánto se acierta sin memoria; debería ser bajo)")
        R.add(S, "LoCoMo · referencia sin memoria", True, f"{pct(r_ok / len(ref_done))} (n={len(ref_done)})",
              "(observación)", "info", "")
    print("   publicado (Mem0, arXiv 2504.19413; otro modelo y otro juez → solo orden de magnitud): "
          + " · ".join(f"{n} {v:.1f}%" for n, v in PUBLISHED))

    wrong = [it for it in pos if not ok(it) and it["ev"]]
    buckets = {b: 0 for b in ("respuesta", "selección", "recuperación", "ingesta")}
    for it in wrong:
        it["bucket"] = evidence_bucket(it["ret"]["in_ctx"], it["ret"]["in_cand"], it["ret"]["has_cell"])
        buckets[it["bucket"]] += 1
    print(f"\n   ¿DÓNDE SE PIERDEN LAS RESPUESTAS? ({len(wrong)} fallos con evidencia)")
    print("   " + " · ".join(f"{b} {n}" for b, n in buckets.items()))
    print("   (respuesta = la evidencia llegó al prompt · selección = estaba entre candidatos · recuperación = existía "
          "un hecho de ese bloque · ingesta = ningún hecho de ese bloque)")
    R.add(S, "LoCoMo · fallos por causa", True, " · ".join(f"{b} {n}" for b, n in buckets.items()), "(diagnóstico)",
          "info", "")
    flat = lambda s: " ".join(str(s).split())
    for b in buckets:
        sel = [it for it in wrong if it["bucket"] == b][:4]
        for it in sel:
            print(f"     ✗[{b[:5]}|{CATS[it['cat']][:5]}|{it['ret']['state'][:4]}] «{it['q'][:60]}» "
                  f"ref «{it['gold'][:40]}» → «{flat(it['a'])[:90]}»")
    bad_n = [it for it in neg if inv(it)]
    print(f"\n   ADVERSARIALES donde el agente inventa: {len(bad_n)}/{len(neg)} (auditar a mano)")
    for it in bad_n[:10]:
        print(f"     ✗[{it['ret']['state'][:4]}] «{it['q'][:60]}» trampa «{it['trap'][:30]}» → «{flat(it['a'])[:90]}»")

    # ---- [PARCHE12] auditoría automática: de dónde sale lo inventado y posibles errores de etiqueta de LoCoMo
    src_cnt: dict[str, int] = {}
    for it in bad_n:
        k = trap_source(it["trap"], it["ret"]["lines"], [str(e.get("text", "")) for e in it["ret"]["epi"][:5]])
        src_cnt[k] = src_cnt.get(k, 0) + 1
    own = [it for it in bad_n if it["subj"] and it["ev_spk"] == [it["subj"]]]
    print(f"   origen del dato inventado: " + " · ".join(f"{k} {v}" for k, v in sorted(src_cnt.items(), key=lambda x: -x[1]))
          + f" · evidencia pronunciada por el PROPIO sujeto (posible etiqueta errónea de LoCoMo): {len(own)}")
    R.add(S, "Invenciones: origen (episódica / hechos)", True,
          " · ".join(f"{k} {v}" for k, v in sorted(src_cnt.items(), key=lambda x: -x[1])) or "—", "(diagnóstico)",
          "info", "agente.py · _build_system", f"posibles etiquetas erróneas: {len(own)}")

    base_path = os.getenv("HMEM_LOCOMO_BASE", "").strip()
    if base_path and Path(base_path).exists():
        try:
            base_rows = [json.loads(ln) for ln in Path(base_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception as exc:
            base_rows = []
            print(f"   ⚠ no se pudo leer la base {base_path}: {exc}")
        new_rows = [{"sid": it["sid"], "q": it["q"], "cat": it["cat"], "veredicto": it.get("v")} for it in main_items]
        cmp = compare_runs(base_rows, new_rows) if base_rows else {}
        if cmp:
            print(f"\n   COMPARACIÓN PAREADA con {Path(base_path).name} (mismas preguntas, mismo contexto recuperado)")
            for key, label, unit in (("pos", "correctas", "bien"), ("neg", "inventa", "inventa")):
                c = cmp[key]
                if not c["n"]:
                    continue
                before = c["antes"] / c["n"] if key == "pos" else 1 - c["antes"] / c["n"]
                after = c["ahora"] / c["n"] if key == "pos" else 1 - c["ahora"] / c["n"]
                worse, better = (c["empeora"], c["mejora"])
                txt = (f"{pct(before)} → {pct(after)} (n={c['n']} · mejoran {better} · empeoran {worse} · "
                       f"p={c['p']:.3f})")
                print(f"   {label:<10} {txt}")
                R.add(S, f"Comparado con la base · {label}", True, txt, "(decisión)", "info", "agente.py · BASE_PROMPT")
            print("   (p = test de signos exacto sobre las preguntas que cambian; con el juez estocástico, "
                  "algo de cambio es ruido: decide con p < 0,05)")

    # ---- [PARCHE11] ¿se inventa más cuando el contexto no dice de quién es cada hecho?
    def split_rate(xs: list[dict], key: Callable[[dict], bool], good: Callable[[dict], bool]) -> str:
        yes, no = [it for it in xs if key(it)], [it for it in xs if not key(it)]
        r = lambda g: f"{pct(sum(map(good, g)) / len(g))} (n={len(g)})" if g else "— (n=0)"
        return f"sí {r(yes)} · no {r(no)}"

    has_usr = lambda it: it["ph"]["usuario"] > 0
    has_oth = lambda it: it["ph"]["otra"] > 0
    has_sub = lambda it: it["ph"]["sujeto"] > 0
    print("\n   IDENTIDAD DEL SUJETO Y ERRORES  (¿el contexto tenía hechos …?)")
    print(f"   adversariales · inventa si hay hechos de 'el usuario' sin nombre: {split_rate(neg, has_usr, inv)}")
    print(f"   adversariales · inventa si hay hechos de la OTRA persona:          {split_rate(neg, has_oth, inv)}")
    print(f"   adversariales · inventa si hay hechos del sujeto preguntado:       {split_rate(neg, has_sub, inv)}")
    print(f"   con respuesta · correctas si hay hechos de 'el usuario' sin nombre: {split_rate(pos, has_usr, ok)}")
    print("   (correlación, no causa: si 'sí' es mucho peor que 'no', la pérdida de identidad en la ingesta es sospechosa)")
    R.add(S, "Inventa con / sin hechos de 'el usuario' en contexto", True, split_rate(neg, has_usr, inv),
          "(diagnóstico)", "info", "ltm/ingestion.py · extract_facts")
    rk_wrong = [it["ret"]["ev_rank"] for it in pos if not ok(it) and it["ev"]]
    mid_wrong = sum(1 for x in rk_wrong if x and x > 6)
    print(f"   fallos con la evidencia en los puestos 7-18 (un top-K mayor podría ayudar): {mid_wrong}/{len(rk_wrong)}")
    R.add(S, "Fallos con evidencia en puestos 7-18", True, f"{mid_wrong}/{len(rk_wrong)}", "(diagnóstico)", "info",
          "ltm/retrieval.py · modo directo (top-6)")

    # detalle completo para auditar (fuera de git)
    out = base / f"resultados_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    try:
        with out.open("w", encoding="utf-8") as fo:
            for it in done_items:
                fo.write(json.dumps({"sid": it["sid"], "cat": it["cat"], "ref": bool(it.get("ref")), "q": it["q"],
                                     "gold": it["gold"], "trap": it["trap"], "answer": it.get("a", ""),
                                     "veredicto": it.get("v"), "estado": it["ret"]["state"],
                                     "evid_prompt": it["ret"]["in_ctx"], "evid_cand": it["ret"]["in_cand"],
                                     "causa": it.get("bucket"), "ms": round(it["ret"]["ms"], 1),
                                     # [PARCHE11] contexto completo para auditar a mano
                                     "sujeto": it["subj"], "hablante_evid": it["ev_spk"],  # [PARCHE12]
                                     "identidad": it["ph"], "top_sim": it["ret"]["top_sim"],
                                     "rango_evid": it["ret"]["ev_rank"], "cobertura": round(it["ret"]["coverage"], 3),
                                     "evidencia": sorted(it["ev"]), "hechos": it["ret"]["lines"],
                                     "episodios": [str(e.get("text", ""))[:300] for e in it["ret"]["epi"][:5]]},
                                    ensure_ascii=False) + "\n")
        print(f"\n   detalle de cada pregunta: {out}")
    except OSError:
        pass

    ns = north_star
    if pos:
        R.add(S, "OBJETIVO LoCoMo · correctas (categorías 1-4)", c_ok / len(pos) >= ns["correctas"],
              f"{pct(c_ok / len(pos))} [{pct(c_lo)}–{pct(c_hi)}]", f"≥ {pct(ns['correctas'])}", "alto",
              "ver 'fallos por causa'", f"n={len(pos)} · comparable en orden de magnitud con el J publicado")
    if not neg:
        return
    R.add(S, "OBJETIVO LoCoMo · inventa (adversariales)", n_inv / max(1, len(neg)) <= ns["inventa"],
          f"{pct(n_inv / max(1, len(neg)))} [{pct(i_lo)}–{pct(i_hi)}]", f"≤ {pct(ns['inventa'])}", "alto",
          "agente.py · BASE_PROMPT / ltm/retrieval.py · compuerta", f"n={len(neg)}",
          [f"«{it['q'][:50]}» → «{flat(it['a'])[:60]}»" for it in bad_n[:4]])
