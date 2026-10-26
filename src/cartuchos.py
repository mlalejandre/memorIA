# -*- coding: utf-8 -*-
"""
src/cartuchos.py — Gestor Multi-Cartucho con soporte para múltiples memorias simultáneas,
importación/exportación, clonación y vaciado.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

import config as C
from largo_plazo import MemoryTree, MemLLM


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40].strip("-")
    return s or "cartucho"


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# Triggers que crea el propio sistema. Si anades uno nuevo, incluyelo aqui.
_ALLOWED_TRIGGERS = frozenset({
    "trg_cells_ai", "trg_cells_ad", "trg_cells_au", "trg_cells_valid_from_ai",
    "trg_emb_cells_au", "trg_emb_cells_ad", "trg_emb_nodes_au", "trg_emb_nodes_ad",
})
MAX_IMPORT_BYTES = int(os.getenv("HMEM_MAX_IMPORT_MB", "2048")) * 2**20


def _schema_problem(db_path: Path) -> str | None:
    """Rechaza bases de datos ajenas con vistas o triggers desconocidos (posible codigo hostil)."""
    try:
        conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True, timeout=5.0)
        try:
            rows = conn.execute("SELECT type, name FROM sqlite_master WHERE type IN ('trigger', 'view')").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return f"La base de datos SQLite no es valida: {exc}"
    bad = [f"{t} {n}" for t, n in rows if t == "view" or n not in _ALLOWED_TRIGGERS]
    if bad:
        return "La base de datos contiene objetos no permitidos: " + ", ".join(bad[:5])
    return None


class CartridgeManager:
    def _dir(self, cid: str) -> Path:
        """Ruta de un cartucho; rechaza ids con '..', separadores o que escapen de root."""
        if not isinstance(cid, str) or not _ID_RE.match(cid):
            raise ValueError(f"Identificador de cartucho inválido: {cid!r}")
        p = self.root / cid
        if p.resolve().parent != self.root.resolve():
            raise ValueError(f"Identificador de cartucho fuera del directorio de cartuchos: {cid!r}")
        return p
    def __init__(self, agent: Any = None, root: Path | None = None) -> None:
        self.agent = agent
        self.root = Path(root or getattr(C, "CARTRIDGES_DIR", C.BODY / "cartuchos"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / ".active.json"
        self._ensure_principal()

    def _ensure_principal(self) -> None:
        principal_dir = self.root / "principal"
        body_dir = getattr(C, "BODY", self.root.parent)
        if not principal_dir.exists():
            principal_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "id": "principal",
                "name": "Memoria Principal",
                "description": "Cartucho inicial con ontología activa y conocimiento consolidado",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (principal_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            orig_db = getattr(C, "LTM_DIR", Path(body_dir) / "largo_plazo") / "memoria.db"
            if orig_db.exists():
                shutil.copy2(orig_db, principal_dir / "memoria.db")

            orig_stm = getattr(C, "STM_DIR", Path(body_dir) / "corto_plazo") / "sesion.json"
            if orig_stm.exists():
                shutil.copy2(orig_stm, principal_dir / "sesion.json")

        db_file = principal_dir / "memoria.db"
        if not db_file.exists():
            try:
                t = MemoryTree(principal_dir, MemLLM())
                t.close()
            except Exception:
                pass

        st = self.load_state()
        if not (self.root / st["primary"]).is_dir():
            st["primary"] = "principal"
            if "principal" not in st["mounted"]:
                st["mounted"].append("principal")
            self.save_state(st)

    def load_state(self) -> dict[str, Any]:
        try:
            if not self.state_file.exists():
                init = {"primary": "principal", "mounted": ["principal"]}
                self.save_state(init)
                return init

            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            primary = data.get("primary", "principal")
            mounted = data.get("mounted", [primary])
            valid_mounted = [cid for cid in mounted if (self.root / cid).is_dir()]

            if not valid_mounted:
                valid_mounted = ["principal"]
            if primary not in valid_mounted:
                primary = valid_mounted[0]

            res = {"primary": primary, "mounted": valid_mounted}
            if valid_mounted != mounted or primary != data.get("primary"):
                self.save_state(res)
            return res
        except Exception:
            fallback = {"primary": "principal", "mounted": ["principal"]}
            self.save_state(fallback)
            return fallback

    def save_state(self, state: dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_primary_id(self) -> str:
        return self.load_state()["primary"]

    def get_active_id(self) -> str:
        return self.get_primary_id()

    def get_mounted_ids(self) -> list[str]:
        return self.load_state()["mounted"]

    def get_primary_path(self) -> Path:
        p = self.root / self.get_primary_id()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_cartridge_path(self, cid: str) -> Path:
        return self._dir(cid)

    def get_active_info(self) -> dict[str, Any]:
        pid = self.get_primary_id()
        for c in self.list_cartridges():
            if c["id"] == pid:
                return c
        return {
            "id": pid,
            "name": pid.capitalize(),
            "description": "Cartucho activo",
            "cells": 0,
            "nodes": 0,
            "is_primary": True,
            "is_mounted": True,
            "size_kb": 0,
        }

    def mount(self, cid: str) -> list[str]:
        if not self._dir(cid).is_dir():
            raise FileNotFoundError(f"No existe el cartucho «{cid}»")
        st = self.load_state()
        if cid not in st["mounted"]:
            st["mounted"].append(cid)
            self.save_state(st)
        return st["mounted"]

    def unmount(self, cid: str) -> list[str]:
        st = self.load_state()
        if cid in st["mounted"]:
            if len(st["mounted"]) <= 1:
                raise ValueError("Debe haber al menos un cartucho conectado.")
            if st["primary"] == cid:
                raise ValueError("No puedes desconectar el cartucho primario de escritura. Selecciona otro primero.")
            st["mounted"].remove(cid)
            self.save_state(st)
        return st["mounted"]

    def set_primary(self, cid: str) -> None:
        if not self._dir(cid).is_dir():
            raise FileNotFoundError(f"No existe el cartucho «{cid}»")
        st = self.load_state()
        if cid not in st["mounted"]:
            st["mounted"].append(cid)
        st["primary"] = cid
        self.save_state(st)

    def list_cartridges(self) -> list[dict[str, Any]]:
        st = self.load_state()
        primary_id = st["primary"]
        mounted_ids = set(st["mounted"])
        items = []

        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            manifest_file = folder / "manifest.json"
            data = {}
            if manifest_file.exists():
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                except Exception:
                    data = {}

            db_file = folder / "memoria.db"
            cells_count = 0
            nodes_count = 0

            if self.agent and hasattr(self.agent, "ltm_trees") and folder.name in self.agent.ltm_trees:
                try:
                    tree = self.agent.ltm_trees[folder.name]
                    with tree.lock:
                        cells_count = tree.conn.execute("SELECT COUNT(*) FROM cells WHERE archived = 0").fetchone()[0]
                        nodes_count = tree.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
                except Exception:
                    pass
            elif db_file.exists():
                try:
                    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=2.0)
                    cells_count = conn.execute("SELECT COUNT(*) FROM cells WHERE archived = 0").fetchone()[0]
                    nodes_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
                    conn.close()
                except Exception:
                    pass

            cid = folder.name
            size_bytes = 0
            for f in (db_file, folder / "memoria.db-wal"):
                if f.exists():
                    size_bytes += f.stat().st_size

            items.append({
                "id": cid,
                "name": data.get("name", cid),
                "description": data.get("description", ""),
                "created_at": data.get("created_at", "-"),
                "updated_at": data.get("updated_at", "-"),
                "cells": cells_count,
                "nodes": nodes_count,
                "is_primary": (cid == primary_id),
                "is_mounted": (cid in mounted_ids),
                "size_kb": round(size_bytes / 1024, 1),
            })
        return items

    def create_cartridge(self, name: str, description: str = "") -> dict[str, Any]:
        name = name.strip() or "Nuevo Cartucho"
        base_slug = slugify(name)
        cid = base_slug
        counter = 1
        while (self.root / cid).exists():
            cid = f"{base_slug}-{counter}"
            counter += 1

        cdir = self.root / cid
        cdir.mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest = {
            "id": cid,
            "name": name,
            "description": description.strip(),
            "created_at": now,
            "updated_at": now,
        }
        (cdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            tree = MemoryTree(cdir, MemLLM())
            tree.close()
        except Exception:
            pass

        return {"id": cid, "name": name, "description": description}

    def clone_cartridge(self, source_id: str, new_name: str, new_description: str = "") -> dict[str, Any]:
        src = self._dir(source_id)
        if not src.is_dir():
            raise FileNotFoundError(f"No existe el cartucho fuente: {source_id}")

        if self.agent and hasattr(self.agent, "ltm_trees") and source_id in self.agent.ltm_trees:
            try:
                tree = self.agent.ltm_trees[source_id]
                if hasattr(tree, "lock") and hasattr(tree, "conn"):
                    with tree.lock:
                        tree.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass

        created = self.create_cartridge(new_name, new_description or f"Clon de {source_id}")
        target = self.root / created["id"]

        src_db = src / "memoria.db"
        if src_db.exists():
            for s in ("-wal", "-shm"):
                (target / f"memoria.db{s}").unlink(missing_ok=True)
            shutil.copy2(src_db, target / "memoria.db")

        src_stm = src / "sesion.json"
        if src_stm.exists():
            shutil.copy2(src_stm, target / "sesion.json")

        return created

    def delete_cartridge(self, cid: str) -> dict[str, Any]:
        st = self.load_state()
        if cid == st.get("primary"):
            return {"ok": False, "error": "No puedes eliminar el cartucho primario activo. Selecciona otro como primario primero."}
        if cid in st.get("mounted", []):
            return {"ok": False, "error": "No puedes eliminar un cartucho que está actualmente conectado. Desconéctalo primero."}
        p = self._dir(cid)
        if not p.is_dir():
            return {"ok": False, "error": "Cartucho no encontrado."}

        if self.agent and hasattr(self.agent, "ltm_trees") and cid in self.agent.ltm_trees:
            try:
                tree = self.agent.ltm_trees.pop(cid)
                if hasattr(tree, "close"):
                    tree.close()
            except Exception:
                pass

        shutil.rmtree(p, ignore_errors=True)
        return {"ok": True}

    def reset_cartridge(self, cid: str) -> dict[str, Any]:
        p = self._dir(cid)
        if not p.is_dir():
            return {"ok": False, "error": f"Cartucho «{cid}» no encontrado."}

        if self.agent and hasattr(self.agent, "ltm_trees") and cid in self.agent.ltm_trees:
            try:
                tree = self.agent.ltm_trees[cid]
                if hasattr(tree, "lock"):
                    with tree.lock:
                        tree.close()
                del self.agent.ltm_trees[cid]
            except Exception:
                pass

        for f in (p / "memoria.db", p / "memoria.db-wal", p / "memoria.db-shm", p / "sesion.json", p / "_archivo.jsonl"):
            f.unlink(missing_ok=True)

        new_tree = MemoryTree(p, MemLLM())
        st = self.load_state()

        if cid in st.get("mounted", []):
            if self.agent and hasattr(self.agent, "ltm_trees"):
                self.agent.ltm_trees[cid] = new_tree
                if cid == st.get("primary"):
                    self.agent.ltm = new_tree
                    if hasattr(self.agent, "stm"):
                        self.agent.stm.reset()
                    self.sync_to_legacy(cid)
        else:
            new_tree.close()

        return {"ok": True}

    def export_cartridge_tar(self, cid: str) -> Path:
        p = self._dir(cid)
        if not p.is_dir():
            raise FileNotFoundError(f"No existe el cartucho: {cid}")

        if self.agent and hasattr(self.agent, "ltm_trees") and cid in self.agent.ltm_trees:
            try:
                tree = self.agent.ltm_trees[cid]
                if hasattr(tree, "lock") and hasattr(tree, "conn"):
                    with tree.lock:
                        tree.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass

        body_dir = getattr(C, "BODY", self.root.parent)
        export_dir = Path(body_dir) / "exportaciones"
        export_dir.mkdir(parents=True, exist_ok=True)
        archive_path = export_dir / f"{cid}_{time.strftime('%Y%m%d_%H%M%S')}.cartridge"

        with tarfile.open(archive_path, "w:gz") as tar:
            for item in ("manifest.json", "memoria.db", "sesion.json"):
                fpath = p / item
                if fpath.exists():
                    tar.add(fpath, arcname=item)
        return archive_path

    def import_sqlite_db(self, raw_bytes: bytes, fallback_name: str = "importado") -> dict[str, Any]:
        created = self.create_cartridge(
            fallback_name.capitalize(),
            f"Importado directamente desde base de datos SQLite ({fallback_name})",
        )
        target = self.root / created["id"]
        (target / "memoria.db").write_bytes(raw_bytes)
        _problem = _schema_problem(target / "memoria.db")
        if _problem:
            shutil.rmtree(target, ignore_errors=True)
            return {"ok": False, "error": _problem}

        try:
            t = MemoryTree(target, MemLLM())
            t.close()
        except Exception as e:
            shutil.rmtree(target, ignore_errors=True)
            return {"ok": False, "error": f"La base de datos SQLite no es válida o está dañada: {e}"}

        self.mount(created["id"])
        if self.agent and hasattr(self.agent, "ltm_trees"):
            self.agent.ltm_trees[created["id"]] = MemoryTree(target, MemLLM())

        return {"ok": True, "cartridge": created}

    def import_cartridge_tar(self, archive_path: Path, fallback_name: str = "importado") -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            try:
                with tarfile.open(archive_path, "r:*") as tar:
                    _total = sum(max(0, m.size) for m in tar.getmembers())
                    if _total > MAX_IMPORT_BYTES:
                        raise ValueError(f"Contenido descomprimido de {_total // 2**20} MB (maximo {MAX_IMPORT_BYTES // 2**20} MB)")
                    if hasattr(tarfile, "data_filter"):
                        tar.extractall(tmp_p, filter="data")
                    else:
                        root_p = tmp_p.resolve()
                        for m in tar.getmembers():
                            dest = (tmp_p / m.name).resolve()
                            unsafe = ((not (m.isfile() or m.isdir())) or m.name.startswith(("/", "\\"))
                                      or (dest != root_p and root_p not in dest.parents))
                            if unsafe:
                                raise ValueError(f"Miembro no seguro en el archivo: {m.name}")
                        tar.extractall(tmp_p)
            except Exception as e:
                return {"ok": False, "error": f"Archivo de cartucho ilegible o dañado: {e}"}

            db_file = tmp_p / "memoria.db"
            if not db_file.exists():
                return {"ok": False, "error": "El archivo .cartridge no contiene una memoria.db válida"}

            _problem = _schema_problem(db_file)
            if _problem:
                return {"ok": False, "error": _problem}

            manifest_file = tmp_p / "manifest.json"
            manifest = {}
            if manifest_file.exists():
                try:
                    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                except Exception:
                    manifest = {}

            name = manifest.get("name") or fallback_name.replace(".cartridge", "").capitalize()
            desc = manifest.get("description", "Cartucho importado")
            created = self.create_cartridge(name, desc)
            target = self.root / created["id"]

            shutil.copy2(db_file, target / "memoria.db")
            stm_file = tmp_p / "sesion.json"
            if stm_file.exists():
                shutil.copy2(stm_file, target / "sesion.json")

            self.mount(created["id"])
            if self.agent and hasattr(self.agent, "ltm_trees"):
                self.agent.ltm_trees[created["id"]] = MemoryTree(target, MemLLM())

            return {"ok": True, "cartridge": created}

    def import_cartridge_b64(self, filename: str, content_b64: str) -> dict[str, Any]:
        if not content_b64:
            return {"ok": False, "error": "Contenido del archivo vacío"}

        try:
            raw = base64.b64decode(content_b64)
        except Exception as e:
            return {"ok": False, "error": f"Error decodificando Base64: {e}"}

        stem = Path(filename).stem
        if stem.endswith(".cartridge") or stem.endswith(".tar"):
            stem = Path(stem).stem

        if raw.startswith(b"SQLite format 3\x00"):
            return self.import_sqlite_db(raw, fallback_name=stem)

        with tempfile.NamedTemporaryFile(suffix=".cartridge", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        try:
            return self.import_cartridge_tar(tmp_path, fallback_name=stem)
        finally:
            tmp_path.unlink(missing_ok=True)

    def sync_to_legacy(self, cid: str) -> None:
        # Obsoleto: la UI y el CLI usan ya los cartuchos como unica fuente de verdad.
        return
