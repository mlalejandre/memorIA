# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import unicodedata
from typing import Any

import config as _cfg
from .common import SYSTEM, rrf_weight

try:
    import navegacion as _nav
except ImportError:
    _nav = None


class RetrievalMixin:
    """Motor de búsqueda y recuperación híbrida (Exacta, FTS5 BM25, Embeddings, Árbol y Sinapsis)."""

    _STOP = frozenset((
        "como", "cual", "cuales", "cuando", "donde", "quien", "quienes", "que", "porque", "cuanto",
        "cuantos", "para", "por", "con", "sin", "sobre", "entre", "desde", "hasta", "del", "los", "las",
        "una", "uno", "unos", "unas", "este", "esta", "esto", "ese", "esa", "eso", "mis", "tus", "sus",
        "hay", "son", "fue", "era", "ser", "estar", "tiene", "tengo", "puedes", "puede", "dime",
        "dijiste", "dije", "the", "and", "what", "which", "who", "how", "you", "are", "was", "that",
        "this", "with", "from",
        # Stopwords de sistema y adverbios temporales (no discriminan entre recuerdos personales)
        "usuario", "usuarios", "agente", "agentes", "ahora", "actualmente", "antes", "anteriormente",
        "hoy", "ayer", "manana", "siempre", "nunca", "tambien", "solo", "otro", "otra", "otros", "otras",
    ))

    @classmethod
    def _fts_terms(cls, text: Any, limit: int = 10) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for w in re.findall(r"[^\W_]+", str(text).lower(), flags=re.UNICODE):
            folded = "".join(ch for ch in unicodedata.normalize("NFKD", w) if not unicodedata.combining(ch))
            if len(folded) < 3 or folded in cls._STOP or folded in seen:
                continue
            seen.add(folded)
            out.append(w)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _qstem(term: str) -> str:
        w = "".join(ch for ch in unicodedata.normalize("NFKD", str(term).lower()) if not unicodedata.combining(ch))
        if len(w) > 4 and w.endswith("s"):
            w = w[:-1]
        return w[: max(4, len(w) - 2)] if len(w) >= 6 else w

    def _fts_search(self, terms: list[str], limit: int = 15,
                    include_history: bool = False) -> list[dict[str, Any]]:
        if not terms:
            return []
        query_parts = []
        for t in terms:
            clean = re.sub(r"[^\w]", "", self._qstem(t), flags=re.UNICODE)
            if clean and clean.upper() not in {"AND", "OR", "NOT"}:
                query_parts.append(f'{clean}*')
        if not query_parts:
            return []
        query = " OR ".join(query_parts)
        temporal_filter = "" if include_history else " AND c.valid_until IS NULL AND c.superseded_by IS NULL "
        try:
            rconn = self._get_reader()
            cur = rconn.cursor()
            sql = (
                "SELECT c.*, bm25(cells_fts) AS rank FROM cells_fts "
                "JOIN cells c ON c.id = cells_fts.cell_id "
                "WHERE cells_fts MATCH ? " + temporal_filter +
                "ORDER BY rank ASC LIMIT ?"
            )
            cur.execute(sql, (query, limit))
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []

    def _exact_search(self, question: str, limit: int = 10,
                      include_history: bool = False) -> list[dict[str, Any]]:
        pat = "%" + question.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        visibility = "" if include_history else " AND archived = 0 AND valid_until IS NULL AND superseded_by IS NULL"
        try:
            rconn = self._get_reader()
            cur = rconn.cursor()
            cur.execute("SELECT * FROM cells WHERE lower(text) LIKE ? ESCAPE '\\'" + visibility + " LIMIT ?",
                        (pat, limit))
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []

    def _expand_query(self, question: str) -> list[str]:
        user = (
            "Pregunta del usuario: " + question + "\n"
            "Devuelve hasta 8 palabras clave, sinónimos o términos relacionados que podrían aparecer "
            "en un recuerdo que responda a la pregunta.\n"
            'Devuelve JSON: {"terminos": ["..."]}'
        )
        try:
            obj = self.llm.ask_json(SYSTEM, user)
        except Exception:
            return []
        terms = obj.get("terminos") if isinstance(obj, dict) else None
        return [str(t) for t in terms][:8] if isinstance(terms, list) else []

    def _restore(self, cell_id: str) -> bool:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE cells SET archived = 0, ruta_original = NULL, importance = MAX(importance, 6.0), "
                "updated_tick = ?, last_access_tick = ?, access_count = access_count + 1 "
                "WHERE id = ? AND archived = 1 AND valid_until IS NULL AND superseded_by IS NULL",
                (self.tick, self.tick, cell_id),
            )
            changed = cur.rowcount > 0
            if changed:
                cur.execute(
                    "UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells WHERE cells.node_id = nodes.id "
                    "AND cells.archived = 0) WHERE id = (SELECT node_id FROM cells WHERE id = ?)",
                    (cell_id,),
                )
            self.conn.commit()
        if changed:
            self._bump()
        return changed

    def _synaptic_search(self, seed_node_ids: list[int], limit: int = 6, include_history: bool = False) -> list[dict[str, Any]]:
        if not getattr(_cfg, "SYN_ENABLED", True) or not seed_node_ids:
            return []
        min_w = float(getattr(_cfg, "SYN_MIN_WEIGHT", 1.2))
        marks = ",".join("?" * len(seed_node_ids))
        try:
            rconn = self._get_reader()
            cur = rconn.cursor()
            cur.execute(f"""
                SELECT target_id, MAX(weight) as max_w
                FROM synapses
                WHERE source_id IN ({marks}) AND target_id NOT IN ({marks}) AND weight >= ?
                GROUP BY target_id
                ORDER BY max_w DESC LIMIT 5
            """, seed_node_ids + seed_node_ids + [min_w])
            neighbors = cur.fetchall()
            if not neighbors:
                return []
            target_ids = [r[0] for r in neighbors]
            weights = {r[0]: r[1] for r in neighbors}
            t_marks = ",".join("?" * len(target_ids))
            temporal_filter = "" if include_history else " AND archived = 0 AND valid_until IS NULL AND superseded_by IS NULL"
            cur.execute(f"""
                SELECT * FROM cells
                WHERE node_id IN ({t_marks}) {temporal_filter}
                ORDER BY importance DESC LIMIT ?
            """, target_ids + [limit])
            cells = [dict(r) for r in cur.fetchall()]
            for c in cells:
                c["synapse_w"] = round(weights.get(c["node_id"], 1.0), 2)
            return cells
        except Exception:
            return []

    def _classify_relevance(self, question: str, candidates: list[dict[str, Any]],
                            sim_of: dict[str, float], exact_hits: list[dict[str, Any]]) -> tuple[bool, bool]:
        """Clasifica la relevancia en dos niveles: (found, tentative).
        - (True, False): Alta certeza (exacta, buen solapamiento o sim >= EMB_ABSTAIN_SIM).
        - (True, True): Banda tentativa (sim >= EMB_TENTATIVE_SIM; se inyecta con aviso de precaución).
        - (False, False): Ruido / abstención total (debajo de la banda de ruido).
        """
        if not candidates:
            return False, False
        top = candidates[0]
        top_id = str(top["id"])
        top_sim = sim_of.get(top_id, float(top.get("sim") or 0.0))

        # 1. Coincidencia exacta literal de subcadena
        if exact_hits and any(str(c["id"]) == top_id for c in exact_hits):
            return True, False

        q_stems = {self._qstem(t) for t in self._fts_terms(question)}
        c_stems = {self._qstem(t) for t in self._fts_terms(top.get("text", ""))}
        overlap = len(q_stems & c_stems)

        emb = getattr(self, "emb", None)
        emb_active = emb is not None and emb.available()
        if not emb_active:
            return (overlap >= 1), False

        # 2. Criterios de alta certeza (Tier 1)
        ov2_min = float(getattr(_cfg, "ABSTAIN_OV2_SIM", 0.35))
        ov1_min = float(getattr(_cfg, "ABSTAIN_OV1_SIM", 0.48))
        high_sim = float(getattr(_cfg, "EMB_ABSTAIN_SIM", 0.58))

        if overlap >= 2 and top_sim >= ov2_min:
            return True, False
        if overlap >= 1 and top_sim >= ov1_min:
            return True, False
        if overlap == 0 and top_sim >= high_sim:
            return True, False

        # 3. Banda tentativa (Tier 2 / Modo C: 0.40 <= sim < 0.58)
        tent_min = float(getattr(_cfg, "EMB_TENTATIVE_SIM", 0.40))
        if top_sim >= tent_min:
            return True, True

        return False, False

    def _is_relevant_result(self, question: str, candidates: list[dict[str, Any]],
                            sim_of: dict[str, float], exact_hits: list[dict[str, Any]]) -> bool:
        """Compatibilidad con métodos y suites anteriores."""
        found, _ = self._classify_relevance(question, candidates, sim_of, exact_hits)
        return found

    def recall(self, question: str, reinforce: bool = True, navigate: bool = True, synthesize: bool | None = None) -> dict[str, Any]:
        started_calls = self.llm.calls
        question = (question or "").strip()

        def empty(methods: list[str]) -> dict[str, Any]:
            return {"found": False, "answer": "", "cells": [], "sources": [], "path": "",
                    "calls": self.llm.calls - started_calls, "methods": methods, "candidates": []}

        if not question:
            return empty([])

        include_history = self._wants_history(question)
        # [PARCHE8] los canales con peso 0 NO se calculan: LIKE '%…%' y FTS5 recorren toda la memoria (O(N)) y su
        # resultado se descartaba. Si los embeddings no devuelven nada, se calculan igualmente como respaldo.
        emb_hits: list[dict[str, Any]] = []
        syn_hits: list[dict[str, Any]] = []
        emb = getattr(self, "emb", None)
        if emb is not None:
            try:
                emb_hits = emb.search_cells(question, include_history=include_history)
            except Exception:
                emb_hits = []
        need_exact = (not emb_hits) or rrf_weight("exacta") > 0
        need_lex = (not emb_hits) or rrf_weight("léxica") > 0 or bool(_cfg.MEM_RECALL_EXPAND)
        exact_hits = self._exact_search(question, include_history=include_history) if need_exact else []
        lexical_hits = (self._fts_search(self._fts_terms(question), include_history=include_history)
                        if need_lex else [])
        rankings: dict[str, list[dict[str, Any]]] = {
            "exacta": exact_hits,
            "léxica-FTS5 (BM25)": lexical_hits,
        }
        if emb_hits:
            rankings["embeddings (coseno)"] = emb_hits

        emb_ids = {c["id"] for c in emb_hits}
        top_emb_sim = max((float(c.get("sim", 0.0)) for c in emb_hits), default=0.0)

        # Expansión semántica si no hay aciertos léxicos y la similitud del embedding no es concluyente (< 0.58)
        semantic_ids: set[str] = set()
        if _cfg.MEM_RECALL_EXPAND and (not exact_hits and not lexical_hits) and (len(emb_hits) < 2 or top_emb_sim < 0.58):
            expanded = self._expand_query(question)
            sem_hits = self._fts_search(self._fts_terms(" ".join(expanded)), include_history=include_history) if expanded else []
            if sem_hits:
                rankings["semántica (expansión LLM)"] = sem_hits
                semantic_ids = {c["id"] for c in sem_hits}

        # Activación Propagada Sináptica
        seed_nodes = list(dict.fromkeys(c["node_id"] for c in (emb_hits[:3] + exact_hits[:2] + lexical_hits[:2])))[:5]
        if seed_nodes and getattr(_cfg, "SYN_ENABLED", True) and (rrf_weight("sinapsis") > 0 or not emb_hits):
            try:
                syn_hits = self._synaptic_search(seed_nodes, limit=int(getattr(_cfg, "SYN_SPREAD_LIMIT", 6)), include_history=include_history)
                if syn_hits:
                    rankings["sinapsis (asociativa)"] = syn_hits
            except Exception:
                syn_hits = []

        tree_ids: set[str] = set()
        if navigate and _nav is not None and _nav.enabled() and (rrf_weight("árbol") > 0 or not emb_hits):
            try:
                nav_hits = _nav.navigate(self, question, include_history=include_history)
            except Exception:
                nav_hits = []
            if nav_hits:
                rankings["árbol (navegación)"] = nav_hits
                tree_ids = {c["id"] for c in nav_hits}

        # Fusión Reciprocal Rank Fusion (RRF)
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        sim_of: dict[str, float] = {str(c["id"]): float(c["sim"]) for c in emb_hits if "sim" in c}

        for _name, hits in rankings.items():
            _w = rrf_weight(_name)
            if not emb_hits and not _name.startswith("sinapsis"):
                _w = max(_w, 1.0)
            if _w <= 0:  # canal desactivado: no aporta candidatos al pool
                continue
            for rank, c in enumerate(hits):
                scores[c["id"]] = scores.get(c["id"], 0.0) + _w / (_cfg.RRF_K + rank + 1)
                by_id.setdefault(c["id"], c)

        if not scores:
            return empty([m for m, h in rankings.items() if h] or ["léxica-FTS5 (BM25)"])

        # Preservar la similitud vectorial original en by_id aunque la célula haya entrado por otro canal
        for cid, sim_val in sim_of.items():
            if cid in by_id:
                by_id[cid]["sim"] = sim_val

        ordered = sorted(scores, key=lambda i: scores[i], reverse=True)[:18]
        candidates = [by_id[i] for i in ordered]
        methods = [m for m, h in rankings.items() if h]

        # [PARCHE4] Reranker (juez LLM o cross-encoder): reordena el top-K y decide si algo responde de verdad.
        rr_info = None
        _rr = getattr(self, "reranker", None)
        if _rr is not None and candidates and _rr.enabled():
            rr_info = _rr.rerank(question, candidates)
            if rr_info is not None:
                candidates = rr_info["candidates"]
                ordered = [c["id"] for c in candidates]
                methods.append(f"reranker ({rr_info['mode']})")
        rr_pub = rr_info["public"] if rr_info is not None else None

        # Evaluación de la compuerta de relevancia / abstención
        if rr_info is not None:
            if rr_info["verdict"]:
                is_relevant, is_tentative = True, False
            elif str(getattr(_cfg, "RERANK_NONE", "tentative")).strip().lower() == "abstain":
                is_relevant, is_tentative = False, False
            else:
                is_relevant, is_tentative = True, True
        else:
            is_relevant, is_tentative = self._classify_relevance(question, candidates, sim_of, exact_hits)
        if not is_relevant:
            _r = empty(methods)
            _r["candidates"] = ordered
            _r["rerank"] = rr_pub
            return _r

        # ======================================================================
        # MODO ZERO-LLM DIRECT RETRIEVAL (Inyección Directa de Hechos RRF)
        # ======================================================================
        if synthesize is None:
            synthesize = not bool(getattr(_cfg, "LTM_DIRECT_RETRIEVAL", False))

        if not synthesize and candidates:
            _n_top = max(1, int(getattr(_cfg, "LTM_DIRECT_TOP_K", 6)))   # [PARCHE13] configurable (antes 6)
            # [PARCHE4] con reranker: solo los recuerdos que responden (o 3 dudosos, con aviso)
            if rr_info is not None:
                _n_top = int(rr_info.get("n_relevant") or 0) or 3
            top_cells = candidates[:max(1, min(_n_top, len(candidates)))]
            used_ids = [c["id"] for c in top_cells]
            sources = sorted({c["fuente"] for c in top_cells if c.get("fuente")})
            paths = sorted({self._path_str(c["node_id"]) for c in top_cells})

            fact_lines = []
            for c in top_cells:
                p_str = self._path_str(c["node_id"])
                p_tag = f"[{p_str}] " if p_str else ""
                src_tag = f" (Fuente: {c['fuente']})" if c.get("fuente") else ""
                fact_lines.append(f"• {p_tag}{c['text']}{src_tag}")

            ans_txt = "\n".join(fact_lines)

            restored = 0
            if reinforce:
                used_nodes = list({c["node_id"] for c in top_cells if c.get("node_id")})
                if len(used_nodes) >= 2:
                    for idx_a in range(len(used_nodes)):
                        for idx_b in range(idx_a + 1, len(used_nodes)):
                            self._reinforce_synapse(used_nodes[idx_a], used_nodes[idx_b])
                for c in top_cells:
                    if c.get("archived") and self._restore(c["id"]):
                        restored += 1
                    self._reinforce(c["id"])
                self.stats["recall_hibrido"] = self.stats.get("recall_hibrido", 0) + 1
                self._save_meta()

            return {
                "found": True,
                "tentative": is_tentative,
                "answer": ans_txt,
                "cells": used_ids,
                "sources": sources,
                "path": " | ".join(paths),
                "calls": 0,
                "methods": methods,
                "restored": restored,
                "candidates": ordered,
                "rerank": rr_pub,
                "direct_retrieval": True
            }

        rows = []
        for c in candidates:
            tag = " [archivada]" if c.get("archived") else ""
            if c.get("sim") is not None:
                tag += f" [similitud {float(c['sim']):.2f}]"
            validity = ""
            if c.get("valid_from") or c.get("valid_until"):
                validity = f" [vigencia: {c.get('valid_from') or 'desconocida'} → {c.get('valid_until') or 'actual'}]"
            rows.append(f"[{c['id']}] [imp {float(c['importance']):.0f}] [ruta: {self._path_str(c['node_id'])}]{tag}{validity} {c['text']}")

        prompt = (
            "Responde a la pregunta del usuario basándote EXCLUSIVAMENTE en los recuerdos proporcionados.\n"
            "INSTRUCCIONES OBLIGATORIAS:\n"
            "1. Si los recuerdos contienen la respuesta (incluso con sinónimos o conceptos equivalentes), "
            "debes responder afirmativamente con found=true.\n"
            "2. En 'answer', responde directamente citando OBLIGATORIAMENTE de forma LITERAL "
            "las palabras clave, nombres propios, estructuras anatómicas y mecanismos presentes en el texto del recuerdo.\n"
            "3. En 'used', incluye SIEMPRE la lista con el ID o IDs exactos de los recuerdos utilizados "
            '(ejemplo: ["0000036-ce65d41a2f43"]). NUNCA uses etiquetas ficticias como \'id1\'.\n'
            "4. Solo si ninguno de los recuerdos tiene relación alguna con la pregunta, devuelve found=false y answer=''.\n\n"
            f"Pregunta: {question}\n\n"
            "Recuerdos encontrados:\n" + "\n".join(rows) + "\n\n"
            'Devuelve EXCLUSIVAMENTE un objeto JSON válido con este formato:\n'
            '{"found": true, "answer": "respuesta directa citando términos clave", "used": ["<id_exacto_del_recuerdo>"]}\n'
            'o {"found": false, "answer": "", "used": []}'
        )
        try:
            res_obj = self.llm.ask_json(SYSTEM, prompt)
        except Exception:
            res_obj = {}

        if not isinstance(res_obj, dict) or not res_obj.get("found"):
            if candidates:
                best_cand = candidates[0]
                if self._is_relevant_result(question, [best_cand], sim_of, exact_hits):
                    res_obj = {
                        "found": True,
                        "answer": best_cand["text"],
                        "used": [best_cand["id"]]
                    }
            if not isinstance(res_obj, dict) or not res_obj.get("found"):
                _r = empty(methods)
                _r["candidates"] = ordered
                _r["rerank"] = rr_pub
                return _r

        raw_used = res_obj.get("used", [])
        used_ids = []
        if isinstance(raw_used, list):
            for raw_item in raw_used:
                raw_str = str(raw_item).strip()
                clean_id = raw_str.strip("[]()\"' ")
                if clean_id in by_id:
                    used_ids.append(clean_id)
                else:
                    for bid in by_id:
                        if bid == clean_id or (clean_id and len(clean_id) >= 7 and clean_id in bid):
                            used_ids.append(bid)
                            break

        if not used_ids and candidates and res_obj.get("found"):
            ans_lower = str(res_obj.get("answer", "")).lower()
            best_match = candidates[0]
            for cand in candidates:
                cand_words = [w for w in re.findall(r"\w+", cand["text"].lower()) if len(w) > 3]
                if any(w in ans_lower for w in cand_words):
                    best_match = cand
                    break
            used_ids = [best_match["id"]]

        used_cells = [by_id[i] for i in used_ids if i in by_id]
        ans_txt = str(res_obj.get("answer", "")).strip()
        ans_lower = ans_txt.lower()
        for cand in candidates[:6]:
            if cand["id"] not in used_ids:
                cand_words = [w for w in re.findall(r"\w+", cand["text"].lower()) if len(w) > 4]
                if cand_words:
                    matches = sum(1 for w in cand_words if w in ans_lower)
                    if matches >= 3 or (matches / len(cand_words) >= 0.5):
                        used_cells.append(cand)
                        used_ids.append(cand["id"])

        used_ids = [c["id"] for c in used_cells]
        sources = sorted({c["fuente"] for c in used_cells if c.get("fuente")})
        paths = sorted({self._path_str(c["node_id"]) for c in used_cells})

        restored = 0
        if reinforce:
            used_nodes = list({c["node_id"] for c in used_cells if c.get("node_id")})
            if len(used_nodes) >= 2:
                for idx_a in range(len(used_nodes)):
                    for idx_b in range(idx_a + 1, len(used_nodes)):
                        self._reinforce_synapse(used_nodes[idx_a], used_nodes[idx_b])

            for c in used_cells:
                if c.get("archived") and self._restore(c["id"]):
                    restored += 1
                self._reinforce(c["id"])
            ex_ids = {c["id"] for c in exact_hits}
            lx_ids = {c["id"] for c in lexical_hits}
            if any(i in ex_ids for i in used_ids):
                self.stats["recall_exact"] = self.stats.get("recall_exact", 0) + 1
            if any(i in lx_ids for i in used_ids):
                self.stats["recall_lexico"] = self.stats.get("recall_lexico", 0) + 1
            if any(i in semantic_ids for i in used_ids):
                self.stats["recall_semantica"] = self.stats.get("recall_semantica", 0) + 1
            if any(i in tree_ids for i in used_ids):
                self.stats["recall_arbol"] = self.stats.get("recall_arbol", 0) + 1
            if any(i in emb_ids for i in used_ids):
                self.stats["recall_embeddings"] = self.stats.get("recall_embeddings", 0) + 1
            self.stats["recall_hibrido"] = self.stats.get("recall_hibrido", 0) + 1
            self._save_meta()

        if used_cells:
            for uc in used_cells:
                primary_text = str(uc.get("text", "")).strip()
                if primary_text and primary_text.lower() not in ans_txt.lower():
                    ans_txt = f"{ans_txt} ({primary_text})"

        return {
            "found": True,
            "tentative": is_tentative,
            "answer": ans_txt,
            "cells": used_ids,
            "sources": sources,
            "path": " | ".join(paths),
            "calls": self.llm.calls - started_calls,
            "methods": methods,
            "restored": restored,
            "candidates": ordered,
            "rerank": rr_pub,
        }
