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
#   HMEM_LOCOMO_DESCOMP  1 = diagnóstico de descomposición de preguntas (techo de cobertura; ver [PARCHE19])
#   HMEM_DESCOMP_FRAC    fracción de preguntas de un salto / temporales que entran en ese diagnóstico (def. 0.12)
#   HMEM_LOCOMO_REFINAR  1 = diagnóstico de diversidad por día y 2.ª pasada (techo de cobertura; ver [PARCHE20])
#   [PARCHE22] banco rápido y reproducible:
#   HMEM_ANSWER_TEMP     temperatura de las respuestas del agente (defecto 0: determinista y cacheable)
#   HMEM_JUDGE_URL       servidor del juez (defecto: el de la memoria larga, ocioso mientras el agente responde)
#   HMEM_JUDGE_WORKERS   juicios en paralelo (defecto 2) · HMEM_JUDGE_VOTES votos por juicio (defecto 1, T=0)
#   HMEM_LOCOMO_NOCACHE  1 = ni lee ni escribe la caché de respuestas/juicios (para medir el ruido del servidor)
#   HMEM_FIXED_NOW       «fecha actual» del prompt (defecto 29/10/2026 12:00): sin ella el prompt cambia cada minuto
#   HMEM_E2E_WORKERS     (solo otras suites de prueba.py)
#   [PARCHE24] HMEM_ANSWER_WORKERS respuestas del agente en paralelo (defecto 1) y HMEM_ANSWER_CACHE_PROMPT (defecto 0):
#                        con 1 petición a la vez y sin caché de prompt el agente repite el 100 % de las respuestas (P23)
#   [PARCHE23] razonamiento y determinismo del agente:
#   HMEM_ANSWER_THINK    off | on (=1) | auto: modo de pensamiento del agente (antes, una sonda comprueba que el
#                        servidor lo active por petición). auto = src/pensar.py decide por pregunta [PARCHE24]
#   HMEM_THINK_MAX_TOKENS  límite de tokens (pensamiento + respuesta) con HMEM_ANSWER_THINK=1 (defecto 4096)
#   HMEM_LOCOMO_DETERMINISMO  1 = diagnóstico: cada pregunta se responde 2 veces en 4 configuraciones (1 o 2 en
#                        paralelo · caché de prompt sí/no) y se mide cuántas respuestas salen idénticas; sin juez
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
import descomponer  # [PARCHE19]
import pensar  # [PARCHE24]
import refinar  # [PARCHE20]

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


# ---- [PARCHE15] profundidad de la evidencia (solo recuperación, sin LLM)
DEPTH_GRID = [(10, 6), (10, 10), (10, 20), (20, 10), (20, 20), (30, 30), (50, 50)]   # (hechos K, episodios E)


def deep_ranks(tree: Any, question: str, ev: set[str], state: dict[str, Any], depth: int = 200) -> dict[str, tuple]:
    """Para cada turno de evidencia: (puesto del primer hecho cuyo bloque lo contiene, puesto del episodio exacto),
    en una búsqueda por embeddings sin umbral hasta `depth`. None = no aparece."""
    emb = getattr(tree, "emb", None)
    if emb is None or not ev:
        return {}
    qv = emb.embed_query(question)
    if qv is None:
        return {}
    cells = [str(r) for r, _s in emb.store.search("cell", qv, depth, -1.0, emb.model)]
    epis = [str(r) for r, _s in emb.store.search("episode", qv, depth, -1.0, emb.model)]
    out: dict[str, tuple] = {}
    for d in ev:
        cr = next((i for i, c in enumerate(cells, 1) if d in state["cell_dia"].get(c, [])), None)
        er = next((i for i, e in enumerate(epis, 1) if state["epi_dia"].get(e) == d), None)
        out[d] = (cr, er)
    return out


def covered(ranks: dict[str, tuple], k: int, e: int) -> float:
    """Fracción de los turnos de evidencia alcanzados con K hechos y E episodios."""
    if not ranks:
        return 0.0
    hit = sum(1 for cr, er in ranks.values() if (cr is not None and cr <= k) or (er is not None and er <= e))
    return hit / len(ranks)


def depth_bucket(r: int | None) -> str:
    return "—" if r is None else "1-6" if r <= 6 else "7-10" if r <= 10 else "11-20" if r <= 20 else (
        "21-50" if r <= 50 else "51-200")


# ---- [PARCHE19] descomposición de la pregunta: caché de subconsultas, oráculo y diagnóstico de techo
DECOMP_DIAG_CATS = (1, 3)   # multi-salto e inferencia: todas sus preguntas entran en el diagnóstico


class DecompCache:
    """Subconsultas por pregunta, en disco: el diagnóstico y el experimento usan EXACTAMENTE las mismas.
    Se invalida si cambian el prompt, el máximo de subconsultas o el modelo."""

    def __init__(self, base: Path, tag: str) -> None:
        h = hashlib.sha256()
        for part in (descomponer.DECOMP_SYSTEM, descomponer.DECOMP_PROMPT, str(descomponer.max_subqueries()), tag):
            h.update(str(part).encode("utf-8") + b"\0")
        self.path = Path(base) / f"descomp_{h.hexdigest()[:10]}.json"
        try:
            self.data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}
        self.new = self.hits = 0

    def get(self, llm: Any, question: str) -> tuple[list[str], float]:
        hit = self.data.get(question)
        if isinstance(hit, dict):
            self.hits += 1
            return list(hit.get("subs") or []), float(hit.get("t") or 0.0)
        subs, secs = descomponer.decompose(llm, question)
        self.data[question] = {"subs": subs, "t": round(secs, 3)}
        self.new += 1
        if self.new % 25 == 0:
            self.save()
        return subs, secs

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass


def oracle_subs(gold: str, n: int = 4) -> list[str]:
    """Techo: buscar con la respuesta de referencia (y sus partes si es una lista). Nunca existe en producción."""
    gold = str(gold or "").strip()
    if not gold:
        return []
    parts = [p.strip(" .") for p in re.split(r",|;|\band\b|\by\b", gold) if len(p.strip(" .")) >= 3]
    out: list[str] = []
    for p in [gold] + (parts if len(parts) > 1 else []):
        if p.lower() not in {o.lower() for o in out}:
            out.append(p)
    return out[:n]


def diag_pick(sid: str, question: str, cat: int, frac: float) -> bool:
    """Multi-salto e inferencia siempre; de un salto y temporales, una fracción determinista (por hash)."""
    if cat in DECOMP_DIAG_CATS:
        return True
    if cat not in POS_CATS:
        return False
    h = int(hashlib.sha1(f"{sid}|{question}".encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) < frac * 10000


def decomp_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: {cat, cov_b, cov_d, cov_o, ctx_b, ctx_d, ctx_o, n_subs, t}. b = actual · d = descomposición · o = oráculo."""
    full = lambda x: x >= 0.999

    def group(sel: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sel)
        if not n:
            return {"n": 0}
        g: dict[str, Any] = {"n": n, "descompone": sum(1 for r in sel if r["n_subs"]) / n}
        for v in "bdo":
            g[f"cov_{v}"] = sum(r[f"cov_{v}"] for r in sel) / n
            g[f"full_{v}"] = sum(1 for r in sel if full(r[f"cov_{v}"])) / n
            g[f"ctx_{v}"] = sum(1 for r in sel if r[f"ctx_{v}"]) / n
        for v in "do":
            better = sum(1 for r in sel if full(r[f"cov_{v}"]) and not full(r["cov_b"]))
            worse = sum(1 for r in sel if full(r["cov_b"]) and not full(r[f"cov_{v}"]))
            g[f"mejora_{v}"], g[f"empeora_{v}"], g[f"p_full_{v}"] = better, worse, sign_p(worse, better)
        return g

    ts = sorted(r["t"] for r in rows if r.get("n_subs") is not None)
    return {"n": len(rows), "cats": {str(c): group([r for r in rows if r["cat"] == c]) for c in POS_CATS},
            "simples": group([r for r in rows if r["cat"] in (2, 4)]),
            "t_p50": percentile(ts, 50), "t_p95": percentile(ts, 95)}


DECOMP_RULE = ("GO al experimento si, en multi-salto, la cobertura COMPLETA de la evidencia sube ≥ 5 pp con la "
               "descomposición (mismo presupuesto K+E) con p < 0,05, y en las preguntas de un salto y temporales la "
               "cobertura media no baja más de 2 pp. Si no: NO-GO; si además el oráculo tampoco sube ≥ 5 pp, la "
               "recuperación no es el cuello de botella del multi-salto.")


def decomp_decision(s: dict[str, Any]) -> tuple[str, str]:
    """Regla de decisión del parche 19, fijada ANTES de medir (ver DECOMP_RULE)."""
    mh, simple = s["cats"].get("1") or {"n": 0}, s.get("simples") or {"n": 0}
    if not mh.get("n"):
        return "SIN DATOS", "no hay preguntas multi-salto en el diagnóstico"
    gain = mh["full_d"] - mh["full_b"]
    o_gain = mh["full_o"] - mh["full_b"]
    drop = (simple["cov_b"] - simple["cov_d"]) if simple.get("n") else 0.0
    why = (f"multi-salto completa {mh['full_b'] * 100:.1f} → {mh['full_d'] * 100:.1f} % ({gain * 100:+.1f} pp, "
           f"p={mh['p_full_d']:.3g}) · oráculo {mh['full_o'] * 100:.1f} % ({o_gain * 100:+.1f} pp) · "
           f"un salto+temporal cobertura media {-drop * 100:+.1f} pp")
    if gain >= 0.05 and mh["p_full_d"] < 0.05 and drop <= 0.02:
        return "GO", why
    if o_gain < 0.05:
        return "NO-GO (techo bajo)", why
    return "NO-GO (la descomposición no alcanza el techo)", why


# ---- [PARCHE20] diversidad por día y segunda pasada: caché, resumen por variante y regla de decisión
class RefineCache:
    """Consultas de la 2.ª pasada por (pregunta, contexto de la 1.ª pasada), en disco: el experimento reutiliza las
    del diagnóstico mientras la 1.ª pasada no cambie. Se invalida si cambian el prompt o el modelo."""

    def __init__(self, base: Path, tag: str) -> None:
        h = hashlib.sha256()
        for part in (refinar.REFINE_SYSTEM, refinar.REFINE_PROMPT, str(descomponer.max_subqueries()), tag):
            h.update(str(part).encode("utf-8") + b"\0")
        self.path = Path(base) / f"refinar_{h.hexdigest()[:10]}.json"
        try:
            self.data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}
        self.new = self.hits = 0

    def get(self, llm: Any, question: str, ctx: str) -> tuple[list[str], float]:
        key = question + "\u241e" + hashlib.sha1(ctx.encode("utf-8")).hexdigest()[:16]
        hit = self.data.get(key)
        if isinstance(hit, dict):
            self.hits += 1
            return list(hit.get("subs") or []), float(hit.get("t") or 0.0)
        subs, secs = refinar.second_pass(llm, question, ctx)
        self.data[key] = {"subs": subs, "t": round(secs, 3)}
        self.new += 1
        if self.new % 25 == 0:
            self.save()
        return subs, secs

    save = DecompCache.save


REFINE_VARIANTS = (("a", "diversidad por día"), ("r", "2.ª pasada"), ("ar", "ambas"))
REFINE_RULE = ("Una variante es ELEGIBLE si, en multi-salto, la cobertura COMPLETA de la evidencia sube ≥ 5 pp frente a "
               "la actual (mismo presupuesto K+E) con p < 0,05, y en las preguntas de un salto y temporales la cobertura "
               "media no baja más de 2 pp. GO con la variante MÁS SIMPLE (diversidad < 2.ª pasada < ambas) cuya ganancia "
               "quede a ≤ 2 pp de la mejor elegible. Si ninguna es elegible: NO-GO.")


def refine_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: {cat, cov_b, cov_a, cov_r, cov_ar, cov_o, ses_b, ses_a, n_ev_ses, n_subs, t}."""
    full = lambda x: x >= 0.999

    def group(sel: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sel)
        if not n:
            return {"n": 0}
        g: dict[str, Any] = {"n": n, "consulta": sum(1 for r in sel if r["n_subs"]) / n,
                             "ses_b": sum(r["ses_b"] for r in sel) / n, "ses_a": sum(r["ses_a"] for r in sel) / n,
                             "ev_ses": sum(r["n_ev_ses"] for r in sel) / n}
        for v in ("b", "a", "r", "ar", "o"):
            g[f"cov_{v}"] = sum(r[f"cov_{v}"] for r in sel) / n
            g[f"full_{v}"] = sum(1 for r in sel if full(r[f"cov_{v}"])) / n
        for v in ("a", "r", "ar", "o"):
            better = sum(1 for r in sel if full(r[f"cov_{v}"]) and not full(r["cov_b"]))
            worse = sum(1 for r in sel if full(r["cov_b"]) and not full(r[f"cov_{v}"]))
            g[f"mejora_{v}"], g[f"empeora_{v}"], g[f"p_full_{v}"] = better, worse, sign_p(worse, better)
        return g

    ts = sorted(r["t"] for r in rows)
    return {"n": len(rows), "cats": {str(c): group([r for r in rows if r["cat"] == c]) for c in POS_CATS},
            "simples": group([r for r in rows if r["cat"] in (2, 4)]),
            "t_p50": percentile(ts, 50), "t_p95": percentile(ts, 95)}


def refine_decision(s: dict[str, Any]) -> tuple[str, str | None, str]:
    """Regla del parche 20, fijada ANTES de medir (ver REFINE_RULE). Devuelve (veredicto, variante, motivo)."""
    mh, simple = s["cats"].get("1") or {"n": 0}, s.get("simples") or {"n": 0}
    if not mh.get("n"):
        return "SIN DATOS", None, "no hay preguntas multi-salto en el diagnóstico"
    parts, ok = [], []
    for v, name in REFINE_VARIANTS:
        gain = mh[f"full_{v}"] - mh["full_b"]
        drop = (simple["cov_b"] - simple[f"cov_{v}"]) if simple.get("n") else 0.0
        elig = gain >= 0.05 - 1e-9 and mh[f"p_full_{v}"] < 0.05 and drop <= 0.02 + 1e-9
        parts.append(f"{name} {gain * 100:+.1f} pp (p={mh[f'p_full_{v}']:.3g}; simples {-drop * 100:+.1f} pp)"
                     + (" ✓" if elig else ""))
        if elig:
            ok.append((v, gain))
    why = (f"multi-salto completa actual {mh['full_b'] * 100:.1f} % · " + " · ".join(parts)
           + f" · oráculo {(mh['full_o'] - mh['full_b']) * 100:+.1f} pp")
    if not ok:
        return "NO-GO", None, why
    best = max(g for _v, g in ok)
    pick = next(v for v, _n in REFINE_VARIANTS if any(v == w and g >= best - 0.02 - 1e-9 for w, g in ok))
    return "GO", pick, why


REFINE_ENV = {"a": {"EPI_DIVERSITY": "dia"}, "r": {"LTM_SECOND_PASS": "1"},
              "ar": {"EPI_DIVERSITY": "dia", "LTM_SECOND_PASS": "1"}}


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

def retrieve(tree: Any, question: str, state: dict[str, Any], evidence: set[str],
             subs: list[str] | None = None, t_decomp: float = 0.0, diversify: bool | None = None) -> dict[str, Any]:
    # [PARCHE19] subs = subconsultas (descomposición u oráculo); t_decomp = segundos de la llamada que las generó
    # [PARCHE20] diversify: None = lo que diga la configuración (EPI_DIVERSITY); True/False lo fuerza
    kw = {"extra_queries": list(subs)} if subs else {}
    t0 = time.perf_counter()
    r = tree.recall(question, reinforce=False, synthesize=False, **kw)
    t1 = time.perf_counter()
    ekw = dict(kw, **({} if diversify is None else {"diversify": diversify}))
    epi = tree.search_episodes(question, int(getattr(C, "EPI_LOOKUP_K", 6)), **ekw) or []
    t2 = time.perf_counter()
    cell_dia, epi_dia = state["cell_dia"], state["epi_dia"]
    hit = lambda ids: any(evidence & set(cell_dia.get(str(i), [])) for i in ids)
    hit_epi = lambda eps: any(epi_dia.get(str(e.get("id"))) in evidence for e in eps)
    injected = list(r.get("cells") or []) if r.get("found") else []
    n_epi = max(1, int(getattr(C, "EPI_PROMPT_K", 5)))   # [PARCHE16] antes [:5] fijo, como el agente
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
        "coverage": coverage(evidence, injected, epi[:n_epi], cell_dia, epi_dia),
        "epi_in": epi[:n_epi],   # [PARCHE16] los episodios que muestra _build_system (EPI_PROMPT_K)
        "lines": fact_lines(r.get("answer", "")) if r.get("found") else [],
        "r": r, "epi": epi, "ms": (t2 - t0 + t_decomp) * 1000, "ms_recall": (t1 - t0) * 1000,
        "ms_descomp": t_decomp * 1000, "subs": list(subs or []),   # [PARCHE19]
        "sessions": len({str(state["epi_dia"].get(str(e.get("id")), "?")).split(":")[0] for e in epi[:n_epi]}),  # [PARCHE20]
        "state": "rechaza" if not r.get("found") else ("tentativa" if r.get("tentative") else "firme"),
        "in_ctx": bool(evidence) and (hit(injected) or hit_epi(epi[:n_epi])),
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


# ---- [PARCHE22] banco rápido: caché persistente, huella del servidor y juez determinista
class BenchCache:
    """Respuestas y veredictos en SQLite (experiment/locomo_cache/banco.sqlite). La clave incluye TODO lo que
    determina el resultado (huella del servidor, temperatura, prompt de sistema, pregunta...): si el prompt de una
    pregunta no cambia, su respuesta y su veredicto se reutilizan; si se corta la ejecución, no se pierde nada."""

    def __init__(self, path: Path) -> None:
        import sqlite3
        import threading
        self.lock = threading.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, answer TEXT, t REAL, ts REAL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, verdict INTEGER, ts REAL)")
        self.conn.commit()
        self.hits = {"a": 0, "j": 0}
        self.new = {"a": 0, "j": 0}

    @staticmethod
    def key(*parts: Any) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(str(p).encode("utf-8") + b"\0")
        return h.hexdigest()

    def get_answer(self, key: str) -> tuple[str, float] | None:
        with self.lock:
            r = self.conn.execute("SELECT answer, t FROM answers WHERE key = ?", (key,)).fetchone()
        if r:
            self.hits["a"] += 1
        return (r[0], float(r[1] or 0.0)) if r else None

    def put_answer(self, key: str, answer: str, t: float) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)", (key, answer, t, time.time()))
            self.conn.commit()
            self.new["a"] += 1

    def get_verdict(self, key: str) -> tuple[bool] | None:
        with self.lock:
            r = self.conn.execute("SELECT verdict FROM verdicts WHERE key = ?", (key,)).fetchone()
        if r:
            self.hits["j"] += 1
        return (bool(r[0]),) if r else None

    def put_verdict(self, key: str, verdict: bool) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO verdicts VALUES (?, ?, ?)", (key, int(bool(verdict)), time.time()))
            self.conn.commit()
            self.new["j"] += 1

    def close(self) -> None:
        with self.lock:
            self.conn.close()


def server_fingerprint(base_url: str) -> dict[str, Any]:
    """Modelo, compilación, slots y contexto de un llama-server (GET /props). Si cambia el modelo o la compilación,
    cambian las respuestas: la huella entra en las claves de la caché y se compara con la de la base."""
    root = str(base_url).rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    info: dict[str, Any] = {"url": root}
    try:
        with urllib.request.urlopen(root + "/props", timeout=5) as resp:
            d = json.loads(resp.read().decode("utf-8", errors="replace"))
        gen = d.get("default_generation_settings") or {}
        info.update({"model": d.get("model_path") or d.get("model_alias") or "?", "build": d.get("build_info", "?"),
                     "slots": d.get("total_slots"), "n_ctx": gen.get("n_ctx")})
    except Exception as exc:
        info.update({"model": str(getattr(C, "REMOTE_MODEL_ALIAS", "?")), "build": "?", "error": type(exc).__name__})
    info["id"] = hashlib.sha256(f"{info.get('model')}|{info.get('build')}".encode("utf-8")).hexdigest()[:12]
    return info


def fp_text(fp: dict[str, Any]) -> str:
    model = str(fp.get("model", "?")).rsplit("/", 1)[-1]
    extra = f" · {fp['error']}" if fp.get("error") else ""
    return f"{model} · {fp.get('build', '?')} · slots {fp.get('slots', '?')} · ctx {fp.get('n_ctx', '?')}{extra}"


def judge_once(llm: Any, system: str, user: str, field: str, tries: int = 2) -> bool | None:
    """Juez determinista (temperatura 0, un voto): mismo veredicto para la misma respuesta, cacheable."""
    from ltm.common import extract_json
    for _ in range(tries):
        try:
            txt = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           max_tokens=300, temperature=0.0)
            obj = extract_json(txt)
            if isinstance(obj, dict) and field in obj:
                return bool(obj[field])
        except Exception:
            continue
    return None


# ---- [PARCHE23] determinismo del agente y modo de pensamiento
DET_CONFIGS = ((2, True), (1, True), (2, False), (1, False))   # (respuestas en paralelo, cache_prompt)
THINK_PROBE = ("Un tren sale a las 9:40 y llega a las 13:15. ¿Cuántos minutos dura el viaje? "
               "Responde solo con el número.")


def gen_tag(workers: int, cache_prompt: bool) -> str:
    """[PARCHE24] cómo se generaron las respuestas (entra en la clave de la caché y en la huella del resultado)."""
    return f"w{int(workers)}cp{int(bool(cache_prompt))}"


def answer_key(fp_id: str, temp: float, max_tokens: int, system: str, q: str, think: bool, gen: str) -> str:
    """Clave de una respuesta del agente. Con la generación antigua (2 en paralelo + caché de prompt) la clave es la
    de los parches 22-23, para reutilizar esas respuestas; con otra generación se añade su etiqueta."""
    parts: list[Any] = ["a1", fp_id, temp, max_tokens if think else 350, system, q]
    if think:
        parts.append("think")
    if gen != "w2cp1":
        parts.append(gen)
    return BenchCache.key(*parts)


def think_mode_env() -> str:
    v = os.getenv("HMEM_ANSWER_THINK", "").strip().lower()
    return "auto" if v == "auto" else ("on" if v in {"1", "true", "yes", "si", "sí", "on"} else "off")


def det_name(workers: int, cache_prompt: bool) -> str:
    return f"{workers} en paralelo · caché de prompt {'sí' if cache_prompt else 'no'}"


def determinism_choice(rows: list[dict[str, Any]], min_same: float = 0.95) -> dict[str, Any]:
    """Regla fijada ANTES de medir: de las configuraciones con ≥ 95 % de respuestas idénticas entre dos pasadas, la
    de mayor ritmo (respuestas por minuto). Si ninguna llega, la más determinista, y se avisa."""
    if not rows:
        return {"elegida": None, "determinista": False, "motivo": "sin datos"}
    good = [r for r in rows if r["iguales"] >= min_same]
    if good:
        best = max(good, key=lambda r: r["ritmo"])
        why = f"la más rápida de las que repiten ≥ {min_same:.0%} de las respuestas"
    else:
        best = max(rows, key=lambda r: (r["iguales"], r["ritmo"]))
        why = f"ninguna repite ≥ {min_same:.0%}: la más determinista"
    return {"elegida": best["nombre"], "workers": best["workers"], "cache_prompt": best["cache_prompt"],
            "determinista": bool(good), "iguales": best["iguales"], "ritmo": best["ritmo"], "motivo": why}


def run_determinism(chat: Any, items: list[dict[str, Any]], temp: float, R: Any, S: str, base: Path,
                    fp: dict[str, Any]) -> dict[str, Any]:
    """Cada pregunta 2 veces por configuración (sin caché del banco ni juez): ¿salen respuestas idénticas?"""
    rows: list[dict[str, Any]] = []
    print(f"\n   DETERMINISMO DEL AGENTE: {len(items)} preguntas × 2 pasadas × {len(DET_CONFIGS)} configuraciones "
          f"(T={temp:g})")
    for workers, cp in DET_CONFIGS:
        times: list[float] = []

        def one(it: dict[str, Any], _cp: bool = cp) -> str:
            t_a = time.perf_counter()
            try:
                return chat.chat([{"role": "system", "content": it["system"]}, {"role": "user", "content": it["q"]}],
                                 max_tokens=350, temperature=temp, cache_prompt=_cp)
            except Exception as exc:
                return f"[error LLM: {type(exc).__name__}]"
            finally:
                times.append(time.perf_counter() - t_a)

        t0 = time.time()
        passes = []
        for _ in range(2):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                passes.append(list(ex.map(one, items)))
        el = max(time.time() - t0, 1e-6)
        same = sum(1 for a, b in zip(*passes) if a == b and not a.startswith("[error")) / max(1, len(items))
        errors = sum(a.startswith("[error") for p in passes for a in p)
        row = {"nombre": det_name(workers, cp), "workers": workers, "cache_prompt": cp, "n": len(items),
               "iguales": same, "ritmo": 2 * len(items) / (el / 60), "t_p50": percentile(times, 50),
               "t_p95": percentile(times, 95), "errores": errors, "min": el / 60}
        rows.append(row)
        print(f"   · {row['nombre']:<36} idénticas {pct(same):>6} · {row['ritmo']:5.1f} respuestas/min · "
              f"mediana {row['t_p50']:.2f} s · errores {errors}", flush=True)
        R.add(S, f"Determinismo · {row['nombre']}", True, f"{pct(same)} idénticas · {row['ritmo']:.1f}/min",
              "(diagnóstico)", "info", "llm.py · cache_prompt / HMEM_E2E_WORKERS")
    choice = determinism_choice(rows)
    print(f"   ⇒ configuración elegida: {choice['elegida']} ({choice['motivo']})")
    R.add(S, "Determinismo · elegida", True, str(choice["elegida"]), "(decisión)", "info", "", choice["motivo"])
    out = {"fecha": time.strftime("%Y-%m-%dT%H:%M:%S"), "agente": fp, "temp": temp, "filas": rows, "decision": choice}
    try:
        (base / f"determinismo_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return out


def probe_verdict(on: dict[str, Any], off: dict[str, Any]) -> bool:
    """¿El servidor piensa cuando se le pide? Razonamiento separado, o muchos más tokens que sin pensar."""
    if not on or on.get("error"):
        return False
    if (on.get("reasoning_chars") or 0) > 0:
        return True
    a, b = on.get("completion_tokens"), off.get("completion_tokens") if off else None
    return bool(a and b and a >= 3 * b + 40)


def thinking_probe(llm: Any, max_tokens: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, flag in (("on", True), ("off", False)):
        t0 = time.perf_counter()
        try:
            txt = llm.chat([{"role": "user", "content": THINK_PROBE}], max_tokens=max_tokens, temperature=0.0,
                           thinking=flag)
            u = llm.last_usage if hasattr(llm, "last_usage") else {}
            out[name] = {"answer": str(txt)[:80], "s": round(time.perf_counter() - t0, 2),
                         "completion_tokens": u.get("completion_tokens"), "reasoning_chars": u.get("reasoning_chars", 0)}
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    out["available"] = probe_verdict(out.get("on", {}), out.get("off", {}))
    out["always_on"] = (out.get("off", {}).get("reasoning_chars") or 0) > 0
    return out


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


def report_refine(rows: list[dict[str, Any]], R: Any, S: str, base: Path) -> dict[str, Any]:
    """[PARCHE20] Imprime el diagnóstico de diversidad / 2.ª pasada, lo guarda en JSON y aplica la regla."""
    s = refine_summary(rows)
    print(f"\n   DIVERSIDAD POR DÍA Y 2.ª PASADA · techo de cobertura sin LLM de respuesta ({len(rows)} preguntas con "
          f"evidencia; mismo presupuesto: {getattr(C, 'LTM_DIRECT_TOP_K', 10)} hechos + {getattr(C, 'EPI_PROMPT_K', 20)} "
          f"episodios; diversidad ≤ {refinar.per_day()} episodios/día)")
    print(f"   llamada de la 2.ª pasada: P50 {s['t_p50'] * 1000:.0f} ms · P95 {s['t_p95'] * 1000:.0f} ms")
    print(f"   {'categoría':<14}{'n':>5}{'ses.ev':>7}{'ses. act→div':>14}   "
          f"{'completa  act / div / 2ª / ambas / orác':>44}   {'cob. media act / div / 2ª / ambas':>36}")
    for name, g in [(CATS[c], s["cats"][str(c)]) for c in POS_CATS] + [("un salto+temp.", s["simples"])]:
        if not g.get("n"):
            continue
        print(f"   {name:<14}{g['n']:>5}{g['ev_ses']:>7.1f}{g['ses_b']:>8.1f}→{g['ses_a']:<5.1f}   "
              + "".join(f"{pct(g[f'full_{v}']):>9}" for v in ("b", "a", "r", "ar", "o")) + "   "
              + "".join(f"{pct(g[f'cov_{v}']):>9}" for v in ("b", "a", "r", "ar")))
    mh = s["cats"].get("1", {})
    if mh.get("n"):
        print("   multi-salto, completa frente a la actual (+/−, p): " + " · ".join(
            f"{name} +{mh[f'mejora_{v}']}/−{mh[f'empeora_{v}']} p={mh[f'p_full_{v}']:.3g}" for v, name in REFINE_VARIANTS))
    verdict, variant, why = refine_decision(s)
    print(f"   REGLA (fijada antes de medir): {REFINE_RULE}")
    print(f"   ⇒ {verdict}{' · ' + dict(REFINE_VARIANTS)[variant] if variant else ''}: {why}")
    for r in [r for r in rows if r["cat"] == 1 and r["n_subs"]][:5]:
        print(f"     «{r['q'][:60]}» → " + " | ".join(x[:40] for x in r["subs"])
              + f"   (completa {r['cov_b']:.2f} → {r['cov_r']:.2f})")
    R.add(S, "Diversidad / 2.ª pasada · decisión", True, verdict + (f" ({variant})" if variant else ""),
          "(regla previa)", "info", "refinar.py", why)
    out = Path(base) / f"refinar_resumen_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        out.write_text(json.dumps({"fecha": time.strftime("%Y-%m-%dT%H:%M:%S"), "regla": REFINE_RULE,
                                   "decision": verdict, "variante": variant,
                                   "variante_nombre": dict(REFINE_VARIANTS).get(variant),
                                   "env": REFINE_ENV.get(variant, {}), "motivo": why, "resumen": s,
                                   "config": {"K": getattr(C, "LTM_DIRECT_TOP_K", 10),
                                              "E": getattr(C, "EPI_PROMPT_K", 20), "por_dia": refinar.per_day(),
                                              "pool": getattr(C, "EPI_DIVERSITY_POOL", 3),
                                              "max_consultas": descomponer.max_subqueries()},
                                   "filas": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"   detalle del diagnóstico: {out}")
    except OSError:
        pass
    return s


def report_decomp(rows: list[dict[str, Any]], R: Any, S: str, base: Path) -> dict[str, Any]:
    """[PARCHE19] Imprime el diagnóstico de descomposición, lo guarda en JSON y aplica la regla de decisión."""
    s = decomp_summary(rows)
    print(f"\n   DESCOMPOSICIÓN DE LA PREGUNTA · techo de cobertura sin LLM de respuesta ({len(rows)} preguntas con "
          f"evidencia; mismo presupuesto: {getattr(C, 'LTM_DIRECT_TOP_K', 10)} hechos + "
          f"{getattr(C, 'EPI_PROMPT_K', 20)} episodios)")
    print(f"   llamada de descomposición: P50 {s['t_p50'] * 1000:.0f} ms · P95 {s['t_p95'] * 1000:.0f} ms")
    print(f"   {'categoría':<14}{'n':>5}{'descomp.':>9}   {'cobertura media  act/desc/orác':>32}   "
          f"{'completa  act/desc/orác':>27}   {'desc vs act (+/−, p)':>22}")
    for name, g in [(CATS[c], s["cats"][str(c)]) for c in POS_CATS] + [("un salto+temp.", s["simples"])]:
        if not g.get("n"):
            continue
        print(f"   {name:<14}{g['n']:>5}{pct(g['descompone']):>9}   "
              f"{pct(g['cov_b']):>10}{pct(g['cov_d']):>8}{pct(g['cov_o']):>8}      "
              f"{pct(g['full_b']):>8}{pct(g['full_d']):>8}{pct(g['full_o']):>8}   "
              f"+{g['mejora_d']}/−{g['empeora_d']} p={g['p_full_d']:.3g}")
    verdict, why = decomp_decision(s)
    print(f"   REGLA (fijada antes de medir): {DECOMP_RULE}")
    print(f"   ⇒ {verdict}: {why}")
    ex = [r for r in rows if r["cat"] == 1 and r["n_subs"]][:6]
    for r in ex:
        print(f"     «{r['q'][:60]}» → " + " | ".join(x[:40] for x in r["subs"])
              + f"   (completa {r['cov_b']:.2f} → {r['cov_d']:.2f})")
    mh = s["cats"].get("1", {})
    if mh.get("n"):
        R.add(S, "Descomposición · multi-salto cobertura completa (act → desc / oráculo)", True,
              f"{pct(mh['full_b'])} → {pct(mh['full_d'])} / {pct(mh['full_o'])}", "(diagnóstico)", "info",
              "descomponer.py", f"p={mh['p_full_d']:.3g} · descompone {pct(mh['descompone'])}")
    R.add(S, "Descomposición · decisión", True, verdict, "(regla previa)", "info", "descomponer.py", why)
    out = Path(base) / f"descomp_resumen_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        out.write_text(json.dumps({"fecha": time.strftime("%Y-%m-%dT%H:%M:%S"), "regla": DECOMP_RULE,
                                   "decision": verdict, "motivo": why, "resumen": s,
                                   "config": {"K": getattr(C, "LTM_DIRECT_TOP_K", 10),
                                              "E": getattr(C, "EPI_PROMPT_K", 20),
                                              "max_subconsultas": descomponer.max_subqueries()},
                                   "filas": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"   detalle del diagnóstico: {out}")
    except OSError:
        pass
    return s


# ==============================================================================
# 5. SUITE
# ==============================================================================

def run(ctx: Any, open_tree: Callable[[Path], Any], sync_all: Callable[[Any], None],
        north_star: dict[str, float]) -> None:
    R, S = ctx.report, "locomo"
    os.environ.setdefault("HMEM_FIXED_NOW", "29/10/2026 12:00")   # [PARCHE22] prompt estable → cacheable
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
    workers = max(1, int(os.getenv("HMEM_ANSWER_WORKERS", "1")))   # [PARCHE24] 1 = agente reproducible
    reingest = os.getenv("HMEM_LOCOMO_REINGEST", "").strip().lower() in {"1", "true", "yes", "si", "sí"}
    models = (str(getattr(C, "REMOTE_MODEL_ALIAS", "")), str(getattr(C, "EMB_MODEL_ALIAS", "")),
              "mock" if ctx.mock else "live")
    base = cache_root() if ctx.live else Path(ctx.tmp) / "locomo"
    print(f"   datos: {info}")
    print(f"   conversaciones: {', '.join(s['sample_id'] for s in samples)} · bloque {size} turnos · "
          f"hechos inyectados (LTM_DIRECT_TOP_K) {getattr(C, 'LTM_DIRECT_TOP_K', 6)} · episodios en el prompt "
          f"(EPI_PROMPT_K) {getattr(C, 'EPI_PROMPT_K', 5)} de {getattr(C, 'EPI_LOOKUP_K', 6)} buscados · caché: {base}")  # [PARCHE13] [PARCHE16]

    items: list[dict[str, Any]] = []
    ingest_stats: list[dict[str, Any]] = []
    deep = os.getenv("HMEM_LOCOMO_PROFUNDIDAD", "").strip().lower() in {"1", "true", "yes", "si", "sí"}  # [PARCHE15]
    only_retrieval = os.getenv("HMEM_LOCOMO_SOLO_RECUPERACION", "").strip().lower() in {"1", "true", "yes", "si", "sí"}
    depth_rows: list[dict[str, Any]] = []
    all_facts: list[str] = []
    # [PARCHE19] descomposición: en producción si LTM_DECOMPOSE=1; diagnóstico de techo con HMEM_LOCOMO_DESCOMP=1
    decomp_on = bool(getattr(C, "LTM_DECOMPOSE", False))
    decomp_diag = os.getenv("HMEM_LOCOMO_DESCOMP", "").strip().lower() in {"1", "true", "yes", "si", "sí"}
    decomp_frac = min(1.0, max(0.0, float(os.getenv("HMEM_DESCOMP_FRAC", "0.12"))))
    dcache = DecompCache(base, "|".join(models)) if (decomp_on or decomp_diag) else None
    diag_rows: list[dict[str, Any]] = []
    # [PARCHE20] diversidad por día (EPI_DIVERSITY) y 2.ª pasada (LTM_SECOND_PASS); diagnóstico con HMEM_LOCOMO_REFINAR=1
    refine_on = refinar.second_pass_enabled()
    refine_diag = os.getenv("HMEM_LOCOMO_REFINAR", "").strip().lower() in {"1", "true", "yes", "si", "sí"}
    rcache = RefineCache(base, "|".join(models)) if (refine_on or refine_diag) else None
    plain = not (decomp_on or refine_on or refinar.diversity_mode() == "dia")   # la recuperación es la actual
    ref_rows: list[dict[str, Any]] = []
    if rcache is not None or refinar.diversity_mode() != "off":
        print(f"   diversidad de episodios: {refinar.diversity_mode()} (≤ {refinar.per_day()} por día) · 2.ª pasada: "
              f"{'ACTIVA' if refine_on else 'inactiva'}{' · diagnóstico de techo' if refine_diag else ''}"
              + (f" · caché {rcache.path.name} ({len(rcache.data)} entradas)" if rcache is not None else ""))
    if dcache is not None:
        print(f"   descomposición: {'ACTIVA en la recuperación' if decomp_on else 'inactiva en la recuperación'}"
              f"{' · diagnóstico de techo' if decomp_diag else ''} · máx. {descomponer.max_subqueries()} "
              f"subconsultas · caché {dcache.path.name} ({len(dcache.data)} preguntas)")
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
            subs, t_d = dcache.get(tree.llm, str(q["question"])) if decomp_on else ([], 0.0)   # [PARCHE19]
            ret = retrieve(tree, str(q["question"]), state, ev, subs, t_d)
            if refine_on:   # [PARCHE20] producción: 2.ª pasada con lo recuperado en la 1.ª
                more, t_r = rcache.get(tree.llm, str(q["question"]), refinar.contexto(ret["lines"], ret["epi_in"]))
                ms_first = ret["ms"]
                if more:
                    ret = retrieve(tree, str(q["question"]), state, ev, subs + more, t_r)
                    ret["ms"] += ms_first
                else:
                    ret["ms"] += t_r * 1000
                ret["subs_ref"], ret["ms_refinar"] = more, t_r * 1000
            if refine_diag and ev and diag_pick(sid, str(q["question"]), cat, decomp_frac):   # [PARCHE20]
                qq = str(q["question"])
                rb = ret if plain else retrieve(tree, qq, state, ev, diversify=False)
                ra = retrieve(tree, qq, state, ev, diversify=True)
                more, t_r = rcache.get(tree.llm, qq, refinar.contexto(rb["lines"], rb["epi_in"]))
                rr = retrieve(tree, qq, state, ev, more, diversify=False) if more else rb
                rar = retrieve(tree, qq, state, ev, more, diversify=True) if more else ra
                o_subs = oracle_subs(str(q.get("answer", "")))
                ro = retrieve(tree, qq, state, ev, o_subs, diversify=False) if o_subs else rb
                ref_rows.append({"sid": sid, "q": qq, "cat": cat, "gold": str(q.get("answer", "")),
                                 "subs": more, "n_subs": len(more), "t": round(t_r, 3),
                                 "n_ev_ses": len({d.split(":")[0] for d in ev}),
                                 "ses_b": rb["sessions"], "ses_a": ra["sessions"],
                                 "cov_b": rb["coverage"], "cov_a": ra["coverage"], "cov_r": rr["coverage"],
                                 "cov_ar": rar["coverage"], "cov_o": ro["coverage"]})
            if decomp_diag and ev and diag_pick(sid, str(q["question"]), cat, decomp_frac):   # [PARCHE19]
                d_subs, d_t = (subs, t_d) if decomp_on else dcache.get(tree.llm, str(q["question"]))
                rb = retrieve(tree, str(q["question"]), state, ev) if decomp_on else ret
                rd = retrieve(tree, str(q["question"]), state, ev, d_subs, d_t) if d_subs else rb
                o_subs = oracle_subs(str(q.get("answer", "")))
                ro = retrieve(tree, str(q["question"]), state, ev, o_subs) if o_subs else rb
                diag_rows.append({"sid": sid, "q": str(q["question"]), "cat": cat, "gold": str(q.get("answer", "")),
                                  "subs": d_subs, "n_subs": len(d_subs), "t": round(d_t, 3),
                                  "cov_b": rb["coverage"], "cov_d": rd["coverage"], "cov_o": ro["coverage"],
                                  "ctx_b": rb["in_ctx"], "ctx_d": rd["in_ctx"], "ctx_o": ro["in_ctx"]})
            spk_a, spk_b = sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]
            subj = subject_of(str(q["question"]), spk_a, spk_b)
            other = (spk_b if subj == spk_a else spk_a) if subj else None
            ev_spk = sorted({spk_of[d] for d in ev if d in spk_of})   # [PARCHE12] quién pronunció la evidencia
            if deep:   # [PARCHE15]
                depth_rows.append({"cat": cat, "ranks": deep_ranks(tree, str(q["question"]), ev, state)})
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
    sys_chars = sorted(len(it["system"]) for it in items)   # [PARCHE16] coste del contexto
    if sys_chars:
        print(f"   prompt de sistema: mediana {percentile(sys_chars, 50) / 1000:.1f} k caracteres · P95 "
              f"{percentile(sys_chars, 95) / 1000:.1f} k (≈ /4 tokens)")
        R.add(S, "Prompt de sistema (mediana / P95, caracteres)", True,
              f"{percentile(sys_chars, 50):.0f} / {percentile(sys_chars, 95):.0f}", "(observación)", "info",
              "agente.py · _build_system")
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

    if deep and depth_rows:   # [PARCHE15] ¿a qué profundidad está la evidencia y qué cobertura daría más contexto?
        rows_ev = [d for d in depth_rows if d["cat"] in POS_CATS and d["ranks"]]
        print(f"\n   PROFUNDIDAD DE LA EVIDENCIA (búsqueda por embeddings sin umbral, {len(rows_ev)} preguntas con evidencia)")
        print("   puesto de cada turno de evidencia · hechos (bloque de 6 turnos) / episodios (turno exacto) · "
              "— = más allá del puesto 200:")
        order = ["1-6", "7-10", "11-20", "21-50", "51-200", "—"]
        for c in POS_CATS:
            sel = [r for d in rows_ev if d["cat"] == c for r in d["ranks"].values()]
            if not sel:
                continue
            ch = {b: sum(1 for cr, _e in sel if depth_bucket(cr) == b) / len(sel) for b in order}
            eh = {b: sum(1 for _c, er in sel if depth_bucket(er) == b) / len(sel) for b in order}
            print(f"   {CATS[c]:<12} n={len(sel):<4} hechos " + " ".join(f"{b}:{pct(ch[b])}" for b in order))
            print(f"   {'':<12} {'':<6} episodios " + " ".join(f"{b}:{pct(eh[b])}" for b in order))
        print("   cobertura media de la evidencia con K hechos + E episodios (techo: el LLM aún tiene que usarla):")
        head = "   " + f"{'categoría':<14}" + "".join(f"{f'K{k}+E{e}':>10}" for k, e in DEPTH_GRID)
        print(head)
        for c in list(POS_CATS) + [0]:
            sel = [d for d in rows_ev if (d["cat"] == c or c == 0)]
            if not sel:
                continue
            name = "todas" if c == 0 else CATS[c]
            vals = [sum(covered(d["ranks"], k, e) for d in sel) / len(sel) for k, e in DEPTH_GRID]
            full = [sum(1 for d in sel if covered(d["ranks"], k, e) >= 0.999) / len(sel) for k, e in DEPTH_GRID]
            print(f"   {name:<14}" + "".join(f"{pct(v):>10}" for v in vals))
            print(f"   {'· completa':<14}" + "".join(f"{pct(v):>10}" for v in full))
            if c in (0, 1):
                R.add(S, f"Cobertura de evidencia por K+E · {name}", True,
                      " · ".join(f"K{k}+E{e} {pct(v)}" for (k, e), v in zip(DEPTH_GRID, vals)), "(diagnóstico)",
                      "info", "ltm/retrieval.py (K) · config.EPI_LOOKUP_K (E)")
        print(f"   (actual: K{getattr(C, 'LTM_DIRECT_TOP_K', 10)}+E{getattr(C, 'EPI_PROMPT_K', 20)} · 'completa' = todas "
              "las evidencias de la pregunta alcanzadas)")  # [PARCHE17]
    if dcache is not None:   # [PARCHE19]
        dcache.save()
        print(f"\n   subconsultas: {dcache.new} generadas ahora · {dcache.hits} reutilizadas de la caché")
    if decomp_on:
        dt = sorted(it["ret"]["ms_descomp"] for it in items)
        n_dec = sum(1 for it in items if it["ret"]["subs"])
        print(f"   descomposición en producción: {n_dec}/{len(items)} preguntas descompuestas · llamada LLM "
              f"P50 {percentile(dt, 50):.0f} ms / P95 {percentile(dt, 95):.0f} ms (incluida en el P95 de recuperación)")
        R.add(S, "Descomposición: preguntas descompuestas / llamada LLM P95", True,
              f"{n_dec}/{len(items)} · {percentile(dt, 95):.0f} ms", "(observación)", "info", "descomponer.py")
    if decomp_diag and diag_rows:
        report_decomp(diag_rows, R, S, base)
    ses = [it["ret"]["sessions"] for it in items if it["cat"] in POS_CATS]   # [PARCHE20]
    if ses:
        print(f"   sesiones distintas entre los episodios del prompt: media {sum(ses) / len(ses):.1f} · "
              f"P10 {percentile(sorted(ses), 10):.0f} · P50 {percentile(sorted(ses), 50):.0f}")
        R.add(S, "Sesiones distintas entre los episodios del prompt (media)", True, f"{sum(ses) / len(ses):.1f}",
              "(observación)", "info", "ltm/explorer.py · search_episodes")
    if rcache is not None:
        rcache.save()
        print(f"   2.ª pasada: {rcache.new} consultas generadas ahora · {rcache.hits} reutilizadas de la caché")
    if refine_on:
        rt = sorted(it["ret"].get("ms_refinar", 0.0) for it in items)
        n_r = sum(1 for it in items if it["ret"].get("subs_ref"))
        print(f"   2.ª pasada en producción: {n_r}/{len(items)} preguntas con consultas nuevas · llamada LLM "
              f"P50 {percentile(rt, 50):.0f} ms / P95 {percentile(rt, 95):.0f} ms (incluida en el P95 de recuperación)")
        R.add(S, "2.ª pasada: preguntas con consultas nuevas / llamada LLM P95", True,
              f"{n_r}/{len(items)} · {percentile(rt, 95):.0f} ms", "(observación)", "info", "refinar.py")
    if refine_diag and ref_rows:
        report_refine(ref_rows, R, S, base)

    if only_retrieval:
        print("   (HMEM_LOCOMO_SOLO_RECUPERACION=1: no se generan respuestas ni se juzga)")
        return

    if not ctx.live:
        print("   ⚠ modo emulado: valida la fontanería (ingesta, mapas de evidencia, prompt); "
              "las respuestas y el juez requieren --live")
        return

    # ---- respuestas reales del agente + juez
    from llm import LLM
    from agente import BASE_PROMPT, now_str
    # [PARCHE22] el agente responde en el servidor STM; el juez, en otro (por defecto el LTM, ocioso en esta fase):
    # responder y juzgar se solapan. Respuestas y juicios deterministas (T=0) y en caché persistente.
    judge_url = os.getenv("HMEM_JUDGE_URL", "").strip() or C.LLM_LTM_URL
    chat = LLM(base_url=C.LLM_STM_URL, model=C.REMOTE_MODEL_ALIAS)
    jchat = LLM(base_url=judge_url, model=C.REMOTE_MODEL_ALIAS)
    fp_a, fp_j = server_fingerprint(C.LLM_STM_URL), server_fingerprint(judge_url)
    temp = float(os.getenv("HMEM_ANSWER_TEMP", "0"))
    votes = max(1, int(os.getenv("HMEM_JUDGE_VOTES", "1")))
    j_workers = max(1, int(os.getenv("HMEM_JUDGE_WORKERS", "2")))
    nocache = os.getenv("HMEM_LOCOMO_NOCACHE", "").strip().lower() in {"1", "true", "yes", "si", "sí"}
    truthy = {"1", "true", "yes", "si", "sí"}
    think_mode = think_mode_env()   # [PARCHE24] off | on | auto
    think = think_mode != "off"
    think_mt = max(512, int(os.getenv("HMEM_THINK_MAX_TOKENS", "4096")))
    mt = think_mt if think else 350
    a_cp = os.getenv("HMEM_ANSWER_CACHE_PROMPT", "0").strip().lower() in truthy   # [PARCHE24]
    gen = gen_tag(workers, a_cp)
    if os.getenv("HMEM_LOCOMO_DETERMINISMO", "").strip().lower() in truthy:   # [PARCHE23]
        run_determinism(chat, items, temp, R, S, base, fp_a)
        return
    if think:
        pr = thinking_probe(chat, mt)
        txt = (f"con pensamiento {pr['on'].get('completion_tokens', '?')} tokens en {pr['on'].get('s', '?')} s "
               f"(razonamiento {pr['on'].get('reasoning_chars', 0)} car.) · sin pensamiento "
               f"{pr['off'].get('completion_tokens', '?')} tokens · respuesta «{pr['on'].get('answer', pr['on'].get('error', ''))}»")
        print(f"\n   SONDA DE PENSAMIENTO: {'disponible' if pr['available'] else 'NO disponible'} · {txt}")
        R.add(S, "Pensamiento del agente (sonda)", True, "disponible" if pr["available"] else "NO disponible",
              "(requisito)", "info", "llm.py · chat_template_kwargs.enable_thinking", txt)
        if not pr["available"]:
            print("   ⚠ el servidor no piensa al pedírselo (¿arrancado sin --jinja, o con --reasoning-budget 0?): "
                  "no se generan respuestas")
            return
        if pr["always_on"]:
            print("   ⚠ el servidor piensa también SIN pedírselo: la base podría no ser «sin pensamiento»")
    bc = BenchCache(base / "banco.sqlite")
    print(f"\n   agente: {fp_text(fp_a)} · T={temp:g} · {workers} a la vez · caché de prompt {'sí' if a_cp else 'no'}"
          + (f" · PENSAMIENTO {think_mode} (máx. {mt} tokens)" if think else ""))
    print(f"   juez:   {fp_text(fp_j)} · {votes} voto(s) · caché {'DESACTIVADA' if nocache else 'activa'} · "
          f"fecha del prompt {now_str()}")
    rnd = random.Random(10)
    ref_items = rnd.sample(pos, min(n_ref, len(pos))) if n_ref else []
    jobs = [(it, it["system"]) for it in items] + [({**it, "ref": True}, BASE_PROMPT.format(now=now_str()))
                                                   for it in ref_items]
    for it, _s in jobs:   # [PARCHE24] ¿piensa esta pregunta? (la referencia sin memoria, nunca)
        it["think"], it["think_why"] = (False, "referencia") if it.get("ref") else pensar.decide(it["q"], think_mode)

    def answer(job: tuple[dict, str]) -> str:
        it, system = job
        th = bool(it.get("think"))
        key = answer_key(fp_a["id"], temp, think_mt, system, it["q"], th, gen)   # [PARCHE24]
        hit = None if nocache else bc.get_answer(key)
        if hit is not None:
            it["t_answer"], it["cached"] = hit[1], True
            return hit[0]
        t_a = time.perf_counter()   # [PARCHE17] tiempo de respuesta del agente (LLM incluido)
        try:
            a = chat.chat([{"role": "system", "content": system}, {"role": "user", "content": it["q"]}],
                          max_tokens=think_mt if th else 350, temperature=temp, cache_prompt=a_cp,
                          **({"thinking": True} if th else {}))
        except Exception as exc:
            it["t_answer"], it["cached"] = time.perf_counter() - t_a, False
            return f"[error LLM: {type(exc).__name__}]"
        it["t_answer"], it["cached"] = time.perf_counter() - t_a, False
        if th:   # [PARCHE23] coste del pensamiento
            it["tokens"] = chat.last_usage.get("completion_tokens")
            it["cut"] = chat.last_finish_reason == "length"
        if not nocache:
            bc.put_answer(key, a, it["t_answer"])
        return a

    def judge(it: dict) -> bool | None:
        if it["cat"] == 5:
            system, field = LOCOMO_NEG_JUDGE, "inventa"
            user = f"Pregunta: {it['q']}\nRespuesta trampa: {it['trap']}\nRespuesta del asistente: {it['a']}"
        else:
            system, field = LOCOMO_POS_JUDGE, "correcta"
            user = f"Pregunta: {it['q']}\nRespuesta de referencia: {it['gold']}\nRespuesta del asistente: {it['a']}"
        if str(it.get("a", "")).startswith("[error LLM"):
            return None
        key = BenchCache.key("j1", fp_j["id"], votes, system, user)
        hit = None if nocache else bc.get_verdict(key)
        if hit is not None:
            return hit[0]
        v = judge_once(jchat, system, user, field) if votes == 1 else _vote(jchat, system, user, field)
        if v is not None and not nocache:
            bc.put_verdict(key, v)
        return v

    t0 = time.time()
    n_jobs = len(jobs)
    print(f"   … {n_jobs} respuestas del agente ({workers} en paralelo) + juez ({j_workers} en paralelo, a la vez)",
          flush=True)
    done_items: list[dict] = [it for it, _s in jobs]

    def progress(n_a: int, n_j: int) -> None:
        el = (time.time() - t0) / 60
        rate = max(n_a + n_j, 1) / max(el, 1e-6)
        left = (2 * n_jobs - n_a - n_j) / rate
        print(f"     respuestas {n_a}/{n_jobs} ({bc.hits['a']} de caché) · juicios {n_j}/{n_jobs} "
              f"({bc.hits['j']} de caché) · {el:.0f} min · quedan ≈{left:.0f} min", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exa, \
            concurrent.futures.ThreadPoolExecutor(max_workers=j_workers) as exj:
        fa = {exa.submit(answer, job): job[0] for job in jobs}
        fj: list[concurrent.futures.Future] = []
        step = max(50, n_jobs // 20)
        for k, f in enumerate(concurrent.futures.as_completed(fa), 1):
            it = fa[f]
            it["a"] = f.result()
            fut = exj.submit(judge, it)
            fut.add_done_callback(lambda ff, _it=it: _it.__setitem__("v", ff.result()))
            fj.append(fut)
            if k % step == 0:
                progress(k, sum(x.done() for x in fj))
        pending = [x for x in fj if not x.done()]   # [PARCHE23] solo los juicios que aún faltan (antes contaba todos)
        done0 = n_jobs - len(pending)
        for k, f in enumerate(concurrent.futures.as_completed(pending), 1):
            if k % step == 0:
                progress(n_jobs, done0 + k)
    for it in done_items:   # la callback puede no haber corrido aún al salir: se asegura el veredicto
        it.setdefault("v", None)
    bc.close()
    print(f"   listo ({(time.time() - t0) / 60:.0f} min) · respuestas nuevas {bc.new['a']} / de caché {bc.hits['a']} · "
          f"juicios nuevos {bc.new['j']} / de caché {bc.hits['j']}")
    R.add(S, "Banco: respuestas y juicios reutilizados de la caché", True,
          f"{bc.hits['a']}/{n_jobs} · {bc.hits['j']}/{n_jobs}", "(observación)", "info", "benchmarks/locomo.py")
    R.add(S, "Servidor del agente / del juez", True, f"{fp_a['id']} / {fp_j['id']}", "(huella)", "info",
          "", f"agente: {fp_text(fp_a)} · juez: {fp_text(fp_j)}")
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
    t_ans = sorted(it["t_answer"] for it in main_items if it.get("t_answer") is not None)   # [PARCHE17]
    if t_ans:
        print(f"\n   tiempo de respuesta del agente (LLM incluido, {workers} en paralelo): mediana "
              f"{percentile(t_ans, 50):.1f} s · P95 {percentile(t_ans, 95):.1f} s")
        R.add(S, "Tiempo de respuesta del agente (mediana / P95)", True,
              f"{percentile(t_ans, 50):.2f} s / {percentile(t_ans, 95):.2f} s", "(observación)", "info",
              "agente.py · _build_system (tamaño del prompt)", f"{workers} en paralelo")
    if think_mode == "auto":   # [PARCHE24] qué decide el modo automático y cuánto cuesta
        routed = [it for it in main_items if it.get("think")]
        why: dict[str, int] = {}
        for it in routed:
            why[it["think_why"]] = why.get(it["think_why"], 0) + 1
        by_cat = " · ".join(f"{CATS[c]} {sum(1 for it in routed if it['cat'] == c)}/{sum(1 for it in main_items if it['cat'] == c)}"
                            for c in sorted({it['cat'] for it in main_items}))
        t_r = [it["t_answer"] for it in routed if it.get("t_answer")]
        t_n = [it["t_answer"] for it in main_items if not it.get("think") and it.get("t_answer")]
        txt = f"{len(routed)}/{len(main_items)} · " + " · ".join(f"{k} {v}" for k, v in sorted(why.items(), key=lambda x: -x[1]))
        print(f"   pensamiento AUTO: piensan {txt} · por categoría: {by_cat}")
        print(f"   tiempo de respuesta: piensan mediana {percentile(t_r, 50):.2f} s · no piensan {percentile(t_n, 50):.2f} s")
        R.add(S, "Pensamiento auto: preguntas que piensan", True, txt, "(observación)", "info", "pensar.py · RULES", by_cat)
        R.add(S, "Pensamiento auto: tiempo piensan / no piensan (mediana)", True,
              f"{percentile(t_r, 50):.2f} s / {percentile(t_n, 50):.2f} s", "(observación)", "info", "pensar.py")
    if think:   # [PARCHE23]
        toks = [it["tokens"] for it in main_items if it.get("tokens")]
        n_cut = sum(1 for it in main_items if it.get("cut"))
        if toks:
            print(f"   pensamiento: {percentile(toks, 50):.0f} tokens por respuesta (mediana) · P95 "
                  f"{percentile(toks, 95):.0f} · cortadas por el límite ({mt}) {n_cut}")
            R.add(S, "Pensamiento: tokens por respuesta (mediana / P95)", True,
                  f"{percentile(toks, 50):.0f} / {percentile(toks, 95):.0f}", "(observación)", "info", "")
        R.add(S, "Pensamiento: respuestas cortadas por el límite", True, f"{n_cut}/{len(main_items)}",
              "(observación)", "info", "HMEM_THINK_MAX_TOKENS")
    a_len = sorted(len(str(it.get("a", ""))) for it in main_items)   # [PARCHE22]
    if a_len:
        print(f"   longitud de la respuesta: mediana {percentile(a_len, 50):.0f} caracteres · P95 {percentile(a_len, 95):.0f}")
        R.add(S, "Longitud de la respuesta (mediana / P95, caracteres)", True,
              f"{percentile(a_len, 50):.0f} / {percentile(a_len, 95):.0f}", "(observación)", "info",
              "agente.py · DIRECT_RULE (ANSWER_DIRECT)")
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
        k = trap_source(it["trap"], it["ret"]["lines"], [str(e.get("text", "")) for e in it["ret"]["epi_in"]])
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
            print(f"\n   COMPARACIÓN PAREADA con {Path(base_path).name} (mismas preguntas; solo cambia lo que prueba el experimento)")
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
            for c in POS_CATS:   # [PARCHE22] comparación pareada por categoría
                cc = compare_runs([r for r in base_rows if int(r["cat"]) == c],
                                  [r for r in new_rows if int(r["cat"]) == c])["pos"]
                if cc["n"]:
                    txt = (f"{pct(cc['antes'] / cc['n'])} → {pct(cc['ahora'] / cc['n'])} (n={cc['n']} · mejoran "
                           f"{cc['mejora']} · empeoran {cc['empeora']} · p={cc['p']:.3f})")
                    print(f"     · {CATS[c]:<11} {txt}")
                    R.add(S, f"Comparado con la base · {CATS[c]}", True, txt, "(decisión)", "info", "")
            if think_mode == "auto":   # [PARCHE24] el efecto, solo donde el modo automático decidió pensar
                routed_rows = [r for r, it in zip(new_rows, main_items) if it.get("think")]
                for key, label in (("pos", "correctas"), ("neg", "inventa")):
                    cc = compare_runs(base_rows, routed_rows)[key]
                    if cc["n"]:
                        a0 = cc["antes"] / cc["n"] if key == "pos" else 1 - cc["antes"] / cc["n"]
                        a1 = cc["ahora"] / cc["n"] if key == "pos" else 1 - cc["ahora"] / cc["n"]
                        txt = (f"{pct(a0)} → {pct(a1)} (n={cc['n']} · mejoran {cc['mejora']} · empeoran "
                               f"{cc['empeora']} · p={cc['p']:.3f})")
                        print(f"     · solo las que piensan · {label:<9} {txt}")
                        R.add(S, f"Comparado con la base · solo las que piensan · {label}", True, txt, "(decisión)",
                              "info", "pensar.py")
            try:   # [PARCHE22] ¿la base se midió con el mismo servidor, juez y forma de medir?
                bm = json.loads(Path(base_path).with_suffix(".meta.json").read_text(encoding="utf-8"))
                same = (bm.get("agente", {}).get("id"), bm.get("juez", {}).get("id"), bm.get("temp"), bm.get("votos"),
                        bm.get("gen", "w2cp1")) == (fp_a["id"], fp_j["id"], temp, votes, gen)   # [PARCHE24] + generación
                msg = "misma huella de servidor, juez y medición" if same else (
                    "⚠ la base se midió con OTRO servidor, juez o forma de medir: la comparación mezcla cambios")
            except Exception:
                same, msg = False, "⚠ la base no registra la huella del servidor (anterior al parche 22)"
            print(f"   {msg}")
            R.add(S, "Base comparable (huella)", True, "sí" if same else "no", "(aviso)", "info", "", msg)
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
    # [PARCHE17] solo una ejecución COMPLETA (10 conversaciones, todas las preguntas) es base de comparaciones
    full = len(samples) == len(data) and not maxq
    stem = f"{'resultados' if full else 'parcial'}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = base / f"{stem}.jsonl"
    for k in range(2, 100):   # [PARCHE22] dos ejecuciones en el mismo segundo no se pisan (la base no se pierde)
        if not out.exists():
            break
        out = base / f"{stem}_{k}.jsonl"
    try:
        with out.open("w", encoding="utf-8") as fo:
            for it in done_items:
                fo.write(json.dumps({"sid": it["sid"], "cat": it["cat"], "ref": bool(it.get("ref")), "q": it["q"],
                                     "gold": it["gold"], "trap": it["trap"], "answer": it.get("a", ""),
                                     "veredicto": it.get("v"), "estado": it["ret"]["state"],
                                     "evid_prompt": it["ret"]["in_ctx"], "evid_cand": it["ret"]["in_cand"],
                                     "causa": it.get("bucket"), "ms": round(it["ret"]["ms"], 1),
                                     "t_respuesta": round(it.get("t_answer") or 0.0, 2),  # [PARCHE17]
                                     "tokens": it.get("tokens"), "cortada": bool(it.get("cut")),  # [PARCHE23]
                                     "piensa": bool(it.get("think")), "motivo_piensa": it.get("think_why"),  # [PARCHE24]
                                     "subconsultas": it["ret"].get("subs") or [],  # [PARCHE19]
                                     "consultas_2a": it["ret"].get("subs_ref") or [],  # [PARCHE20]
                                     "sesiones_prompt": it["ret"].get("sessions"),
                                     # [PARCHE11] contexto completo para auditar a mano
                                     "sujeto": it["subj"], "hablante_evid": it["ev_spk"],  # [PARCHE12]
                                     "identidad": it["ph"], "top_sim": it["ret"]["top_sim"],
                                     "rango_evid": it["ret"]["ev_rank"], "cobertura": round(it["ret"]["coverage"], 3),
                                     "evidencia": sorted(it["ev"]), "hechos": it["ret"]["lines"],
                                     "episodios": [str(e.get("text", ""))[:300] for e in it["ret"]["epi_in"]]},
                                    ensure_ascii=False) + "\n")
        print(f"\n   detalle de cada pregunta: {out}")
        out.with_suffix(".meta.json").write_text(json.dumps({   # [PARCHE22] huella de la medición
            "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"), "agente": fp_a, "juez": fp_j, "temp": temp, "votos": votes,
            "nocache": nocache, "fixed_now": now_str(), "answer_direct": bool(getattr(C, "ANSWER_DIRECT", False)),
            "think": think, "max_tokens": mt, "cache_prompt": bool(getattr(C, "LLM_CACHE_PROMPT", True)),  # [PARCHE23]
            "think_mode": think_mode, "gen": gen, "workers": workers, "answer_cache_prompt": a_cp,  # [PARCHE24]
            "config": {k: str(getattr(C, k, "")) for k in ("LTM_DIRECT_TOP_K", "EPI_PROMPT_K", "EPI_LOOKUP_K",
                                                          "RETRIEVAL_PROFILE", "RERANK_MODE", "LTM_DECOMPOSE",
                                                          "EPI_DIVERSITY", "LTM_SECOND_PASS")}},
            ensure_ascii=False, indent=1), encoding="utf-8")
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
