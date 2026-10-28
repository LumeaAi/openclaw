"""Client de federație: descoperă noduri prin vecini și le înregistrează ca modele."""
from __future__ import annotations

import json
import urllib.request
from typing import Callable

from node import sign
from registry import ModelCard, Registry


def remote_fn(url: str, secret: bytes, timeout: float = 10) -> Callable[[str], str]:
    def call(prompt: str) -> str:
        body = json.dumps({"prompt": prompt}).encode()
        req = urllib.request.Request(url + "/run", body, {"X-Sig": sign(secret, body)})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)["output"]
        except Exception:
            return ""  # nod căzut, refuzat sau lent: un răspuns gol pierde la verificare
    return call


def discover(seeds: list[str], max_nodes: int = 1000) -> dict[str, dict]:
    """Parcurgere în lățime prin lista de vecini a fiecărui nod, pornind de la câteva semințe."""
    found: dict[str, dict] = {}
    queue = list(seeds)
    while queue and len(found) < max_nodes:
        url = queue.pop(0)
        if url in found:
            continue
        try:
            with urllib.request.urlopen(url + "/card", timeout=5) as r:
                card = json.load(r)
        except Exception:
            continue
        found[url] = card
        queue.extend(p for p in card.get("peers", []) if p not in found)
    return found


def join(registry: Registry, cards: dict[str, dict], secret: bytes) -> None:
    for url, c in cards.items():
        registry.add(ModelCard(url, "peer", cost=c["cost"], meta={"name": c["name"]}),
                     remote_fn(url, secret))
