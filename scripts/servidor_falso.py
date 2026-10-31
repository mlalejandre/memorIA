#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Servidor OpenAI-compatible FALSO para probar la rama --live de un parche sin GPU: respuestas aleatorias
pero con forma válida (hechos, juicios, rutas), embeddings por n-gramas, /props y /health.

    python3 scripts/servidor_falso.py 18087 18088 18089 &   y en el .env de la copia de prueba:
    SERVER_IP=127.0.0.1 SERVER_PORT_GPU0=18087 SERVER_PORT_GPU1=18088 SERVER_PORT_GPU2=18089
Con NOTHINK=1 no devuelve razonamiento aunque se le pida.
"""
import hashlib, json, math, os, random, re, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

rnd = random.Random(7)


def vec(t, dim=64):
    v = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", t.lower()):
        for g in ([w] if len(w) < 4 else [w[i:i + 3] for i in range(len(w) - 2)]):
            h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
            v[h % dim] += 1 if (h >> 20) & 1 else -1
    n = math.sqrt(sum(x * x for x in v)) or 1
    return [x / n for x in v]


def answer(user):
    if "ha encontrado una primera búsqueda" in user:
        m = re.findall(r"^- (.+)$", user, re.M)
        return json.dumps({"consultas": [m[-1][:60]] if m else []})
    if "Decide si basta" in user:
        q = re.search(r"Pregunta: (.*)", user).group(1).rstrip("?")
        parts = [p.strip() for p in re.split(r"\band\b", q) if len(p.strip()) > 3]
        return json.dumps({"consultas": parts if len(parts) > 1 else []})
    if "Analiza el siguiente" in user:
        body = re.search(r"<<<(.*?)>>>", user, re.S).group(1)
        lines = [l for l in body.splitlines()[1:] if ":" in l]
        return json.dumps({"hechos": [{"texto": l, "importancia": 6, "tipo": "dato"} for l in lines]})
    if "Dato a memorizar" in user:
        if "- conversaciones (" in user:
            return json.dumps({"use": "conversaciones"})
        if "Profundidad actual: 1" in user:
            return json.dumps({"use": None, "here": True})
        return json.dumps({"use": None, "path": [{"name": "Conversaciones", "description": "x"}]})
    if "CLASIFICA" in user:
        return json.dumps({"relation": "unrelated", "confidence": 0.9})
    if '"correcta"' in user:
        return json.dumps({"correcta": rnd.random() < 0.6})
    if '"inventa"' in user:
        return json.dumps({"inventa": rnd.random() < 0.1})
    return "Respuesta falsa " + str(rnd.random())


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/props"):
            return self._send({"model_path": "/models/fake-%d.gguf" % self.server.server_port, "build_info": "b-test",
                               "total_slots": 2, "default_generation_settings": {"n_ctx": 8192}})
        self._send({"status": "ok"})

    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path.endswith("/embeddings"):
            return self._send({"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(d["input"])]})
        if self.path.endswith("/tokenize"):
            return self._send({"tokens": list(range(max(1, len(d.get("content", "")) // 4)))})
        user = " ".join(m["content"] for m in d["messages"])
        think = bool((d.get("chat_template_kwargs") or {}).get("enable_thinking")) and not os.getenv("NOTHINK")
        msg = {"content": answer(user)}
        if think:
            msg["reasoning_content"] = "pensando " * 50
        self._send({"choices": [{"message": msg, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 400 if think else 12, "prompt_tokens": 100}})


for port in map(int, sys.argv[1:]):
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
threading.Event().wait()
