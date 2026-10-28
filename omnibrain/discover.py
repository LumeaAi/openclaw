"""Descoperire de candidați. Returnează doar fișe: nu descarcă și nu rulează nimic
(un model sau cod necunoscut e risc de securitate până e evaluat în sandbox)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from registry import ModelCard


def _get(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "omnibrain"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def hf_candidates(query: str, limit: int = 20) -> tuple[list[ModelCard], str]:
    url = ("https://huggingface.co/api/models?" + urllib.parse.urlencode(
        {"search": query, "filter": "text-generation", "sort": "downloads",
         "direction": "-1", "limit": limit}))
    try:
        items = _get(url)
    except Exception as e:  # rețea blocată, rate limit etc.
        return [], f"hf: {e}"
    cards = []
    for it in items:
        lic = next((t[8:] for t in it.get("tags", []) if t.startswith("license:")), "unknown")
        cards.append(ModelCard(it["id"], "hf", lic, meta={"downloads": it.get("downloads")}))
    return cards, ""


def github_candidates(query: str, limit: int = 20) -> tuple[list[ModelCard], str]:
    url = ("https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": f"{query} topic:llm", "sort": "stars", "per_page": limit}))
    try:
        items = _get(url)["items"]
    except Exception as e:
        return [], f"github: {e}"
    cards = []
    for it in items:
        lic = (it.get("license") or {}).get("spdx_id") or "unknown"
        cards.append(ModelCard(it["full_name"], "github", lic.lower(),
                               meta={"stars": it.get("stargazers_count")}))
    return cards, ""
