"""Registru de modele din orice sursă: Hugging Face, GitHub, API-uri, local."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Licențe permisive; restul sunt marcate, nu folosite automat.
FREE_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "mpl-2.0"}


@dataclass
class ModelCard:
    id: str
    source: str  # hf | github | api | local | fake
    license: str = "unknown"
    cost: float = 1.0  # cost relativ pe apel
    meta: dict = field(default_factory=dict)

    @property
    def free(self) -> bool:
        return self.license.lower() in FREE_LICENSES


def openai_compatible(base_url: str, model: str, api_key_env: str = "",
                      timeout: int = 60) -> Callable[[str], str]:
    """Un singur adaptor pentru OpenRouter, Together, Groq, Mistral, DeepSeek,
    Ollama, vLLM, LM Studio și orice server cu /chat/completions."""
    def call(prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key_env and os.environ.get(api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[api_key_env]}"
        body = json.dumps({"model": model,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     body, headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    return call


class Registry:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.cards: dict[str, ModelCard] = {}
        self.fns: dict[str, Callable[[str], str]] = {}
        self.path = path
        if path and path.exists():
            for c in json.loads(path.read_text()):
                self.cards[c["id"]] = ModelCard(**c)

    def add(self, card: ModelCard, fn: Optional[Callable[[str], str]] = None) -> None:
        self.cards[card.id] = card
        if fn:
            self.fns[card.id] = fn
        self.save()

    def runnable(self) -> list[str]:
        return [i for i in self.cards if i in self.fns]

    def save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps([asdict(c) for c in self.cards.values()],
                                            indent=1))
