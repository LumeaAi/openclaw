"""Nod federat: expune orice handler (model, agent, API) printr-un protocol HTTP semnat.

GET  /card    public: nume, cost, vecini (pentru descoperire)
POST /run     semnat: {"prompt": str} -> {"output": str}
Cererile sunt semnate HMAC cu secretul federației; fiecare nod are un plafon de apeluri.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


def sign(secret: bytes, body: bytes) -> str:
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


class Node:
    def __init__(self, name: str, handler: Callable[[str], str], secret: bytes,
                 cost: float = 1.0, peers: list[str] | None = None,
                 max_calls: int = 10_000, host: str = "127.0.0.1") -> None:
        self.name, self.handler, self.secret = name, handler, secret
        self.cost, self.peers, self.max_calls = cost, list(peers or []), max_calls
        self.calls = 0
        self._lock = threading.Lock()
        node = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # liniștit
                pass

            def _send(self, code: int, obj: dict) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path != "/card":
                    return self._send(404, {})
                self._send(200, {"name": node.name, "url": node.url,
                                 "cost": node.cost, "peers": node.peers})

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                given = self.headers.get("X-Sig", "")
                if not hmac.compare_digest(given, sign(node.secret, body)):
                    return self._send(403, {"error": "semnătură invalidă"})
                if self.path != "/run":
                    return self._send(404, {})
                with node._lock:
                    if node.calls >= node.max_calls:
                        return self._send(429, {"error": "plafon atins"})
                    node.calls += 1
                prompt = json.loads(body)["prompt"]
                self._send(200, {"output": node.handler(prompt)})

        self.server = ThreadingHTTPServer((host, 0), H)
        self.url = f"http://{host}:{self.server.server_address[1]}"

    def start(self) -> "Node":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
