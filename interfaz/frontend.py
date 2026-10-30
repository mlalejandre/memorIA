# Servidor HTTP local de la consola cognitiva.
from __future__ import annotations

import http.server
import json
import os
import time
import urllib.parse
import webbrowser
from socketserver import ThreadingMixIn

import config as C

HTML_FILE = os.path.join(os.path.dirname(__file__), "index.html")
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_BODY = 20 * 1024 * 1024


class ThreadedHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    # ---------- utilidades ----------
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _origin_ok(self) -> bool:
        """Defensa contra CSRF / DNS-rebinding: Host y Origin locales (IPv4, IPv6 y localhost)."""
        raw_host = (self.headers.get("Host") or "").strip()
        if raw_host.startswith("["):
            end = raw_host.find("]")
            host = raw_host[1:end].lower() if end != -1 else ""
        else:
            host = raw_host.split(":", 1)[0].lower()
        if host not in ALLOWED_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if origin is not None:
            parsed = urllib.parse.urlparse(origin)
            orig_host = (parsed.hostname or "").strip("[]").lower()
            if orig_host not in ALLOWED_HOSTS:
                return False
        return True

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            raise ValueError("Content-Length inválido")
        if length > MAX_BODY:
            raise ValueError("Cuerpo demasiado grande")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _dispatch(self, fn):
        try:
            if not self._origin_ok():
                self._json({"error": "forbidden"}, 403)
                return
            fn()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, TypeError, KeyError) as exc:
            self._send_error(400, exc)
        except Exception as exc:
            self._send_error(500, exc)

    def _send_error(self, code, exc):
        try:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, code)
        except Exception:
            pass

    def do_GET(self):
        self._dispatch(self._get)

    def do_POST(self):
        self._dispatch(self._post)

    def _sse(self, events):
        """Emite un generador de eventos (dict) como Server-Sent Events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for ev in events:
                self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                err = {"event": "error", "error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write(("data: " + json.dumps(err, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass
        finally:
            close = getattr(events, "close", None)
            if close:
                close()

    # ---------- rutas ----------
    def _get(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        agent = self.server.agent
        if path in ("/", "/index.html"):
            with open(HTML_FILE, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._json(agent.status())
        elif path == "/api/stm":
            self._json({"entries": agent.stm.snapshot()})
        elif path == "/api/tree":
            self._json({"tree": agent.ltm.render()})
        elif path in ("/api/graph", "/api/explorer/graph"):
            qs = urllib.parse.parse_qs(parsed.query)
            cid = qs.get("cartridge", [None])[0]
            if hasattr(agent, "get_graph_data"):
                self._json(agent.get_graph_data(cid))
            else:
                fn = getattr(agent.ltm, "get_graph_data", None)
                self._json(fn() if fn else {"nodes": [], "synapses": []})
        elif path == "/api/explorer/tree":
            self._json(agent.ltm.get_explorer_tree())
        elif path == "/api/explorer/cells":
            qs = urllib.parse.parse_qs(parsed.query)
            self._json(agent.ltm.get_node_cells(int(qs.get("node_id", [0])[0])))
        elif path == "/api/episodes":
            qs = urllib.parse.parse_qs(parsed.query)
            q = (qs.get("q", [""])[0]).strip()
            fn = getattr(agent.ltm, "search_episodes", None)
            self._json({"episodes": fn(q, 20) if (q and fn) else []})
        elif path == "/api/cartridges":
            mgr = getattr(agent, "cartridge_mgr", None)
            self._json({
                "cartridges": mgr.list_cartridges() if mgr else [],
                "active": mgr.get_primary_id() if mgr else "principal",
                "mounted": mgr.get_mounted_ids() if mgr else ["principal"]
            })
        elif path == "/api/cartridges/export":
            mgr = getattr(agent, "cartridge_mgr", None)
            qs = urllib.parse.parse_qs(parsed.query)
            cid = qs.get("id", [mgr.get_primary_id() if mgr else "principal"])[0]
            try:
                tar_path = mgr.export_cartridge_tar(cid)
                with open(tar_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Disposition", f'attachment; filename="{cid}.cartridge"')
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                self._send_error(500, exc)
        elif path == "/api/debug":
            self._json(agent.status())
        else:
            self.send_response(404)
            self.end_headers()

    def _post(self):
        path = urllib.parse.urlparse(self.path).path
        agent = self.server.agent
        payload = self._read_json()

        if path == "/api/chat":
            self._json(agent.chat(payload.get("text", ""), payload.get("think")))   # [PARCHE24]
        elif path == "/api/chat/stream":
            self._sse(agent.chat_stream(str(payload.get("text", "")), payload.get("think")))
        elif path == "/api/compact":
            self._json({"reports": agent.stm.compact(force=True)})
        elif path == "/api/reset_stm":
            agent.stm.reset()
            agent.log(">> Memoria a corto plazo vaciada por usuario.")
            self._json({"ok": True})
        elif path == "/api/ingest_ltm":
            text = str(payload.get("text", "")).strip()
            source = str(payload.get("source", "")).strip()
            if not text:
                self._json({"error": "Texto vacío"})
                return
            self._json(agent.ingest_long_text(text, source))
        elif path == "/api/ingest_ltm/stream":
            self._sse(agent.ingest_stream(str(payload.get("text", "")).strip(),
                                          str(payload.get("source", "")).strip()))
        elif path == "/api/explorer/cell/save":
            self._json(agent.ltm.save_cell(
                payload.get("id"), int(payload.get("node_id")), str(payload.get("text", "")),
                float(payload.get("importance", 5)), str(payload.get("kind", "dato")),
                str(payload.get("fuente", "")),
            ))
        elif path == "/api/explorer/cell/delete":
            self._json({"ok": agent.ltm.delete_cell(payload.get("id"))})
        elif path == "/api/explorer/node/save":
            self._json(agent.ltm.save_node(
                payload.get("id"), payload.get("parent_id"),
                str(payload.get("name", "")), str(payload.get("description", "")),
            ))
        elif path == "/api/explorer/node/delete":
            self._json({"ok": agent.ltm.delete_node(int(payload.get("id")))})
        elif path == "/api/cartridges/mount":
            fn = getattr(agent, "mount_cartridge", None)
            self._json(fn(str(payload.get("id", ""))) if fn else {"ok": False})
        elif path == "/api/cartridges/unmount":
            fn = getattr(agent, "unmount_cartridge", None)
            self._json(fn(str(payload.get("id", ""))) if fn else {"ok": False})
        elif path == "/api/cartridges/set_primary":
            fn = getattr(agent, "set_primary_cartridge", None)
            self._json(fn(str(payload.get("id", ""))) if fn else {"ok": False})
        elif path == "/api/cartridges/switch":
            fn = getattr(agent, "set_primary_cartridge", None)
            self._json(fn(str(payload.get("id", "principal"))) if fn else {"ok": True})
        elif path == "/api/cartridges/create":
            mgr = getattr(agent, "cartridge_mgr", None)
            name = str(payload.get("name", "")).strip()
            desc = str(payload.get("description", "")).strip()
            self._json(mgr.create_cartridge(name, desc) if mgr else {"ok": False})
        elif path == "/api/cartridges/clone":
            mgr = getattr(agent, "cartridge_mgr", None)
            src = str(payload.get("source_id", "principal"))
            name = str(payload.get("name", "")).strip()
            desc = str(payload.get("description", "")).strip()
            self._json(mgr.clone_cartridge(src, name, desc) if mgr else {"ok": False})
        elif path == "/api/cartridges/delete":
            mgr = getattr(agent, "cartridge_mgr", None)
            cid = str(payload.get("id", ""))
            self._json(mgr.delete_cartridge(cid) if mgr else {"ok": False})
        elif path == "/api/cartridges/import":
            mgr = getattr(agent, "cartridge_mgr", None)
            fn = str(payload.get("filename", "importado.cartridge"))
            b64 = str(payload.get("content_b64", ""))
            self._json(mgr.import_cartridge_b64(fn, b64) if mgr else {"ok": False})
        elif path == "/api/cartridges/reset":
            cid = payload.get("id")
            fn = getattr(agent, "reset_cartridge", None) or getattr(agent, "reset_active_cartridge", None)
            self._json(fn(cid) if (fn and cid) else fn() if fn else {"ok": False})
        elif path == "/api/explorer/export_disk":
            export_path = C.BODY / f"largo_plazo_exportado_{time.strftime('%Y%m%d_%H%M%S')}"
            n = agent.ltm.export_to_filesystem(export_path)
            self._json({"ok": True, "count": n, "path": str(export_path)})
        else:
            self.send_response(404)
            self.end_headers()


def run_server(agent, port=8765):
    server = None
    actual = port
    for p in range(port, port + 20):
        try:
            server = ThreadedHTTPServer(("127.0.0.1", p), Handler)
            actual = p
            break
        except OSError:
            continue
    if not server:
        raise RuntimeError("No hay puertos disponibles.")
    server.agent = agent
    url = f"http://127.0.0.1:{actual}"
    print(f"[*] Consola cognitiva activa en: {url}")
    if not os.getenv("HMEM_NO_BROWSER"):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
        server.server_close()
