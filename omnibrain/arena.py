"""Arena: modelele concurează pe sarcini verificabile; încrederea se câștigă doar
prin verificare, per categorie, cu penalizare de cost.

Alegerea se face prin Thompson sampling: un model necunoscut e încercat cât să
afle cât valorează, unul dovedit e folosit. Din mii de modele rulează doar cele
alese la fiecare sarcină.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from registry import Registry

VOTE = "vote:top3"


class Arena:
    def __init__(self, registry: Registry, cost_weight: float = 0.05,
                 seed: Optional[int] = None, state_path: Optional[Path] = None) -> None:
        self.reg = registry
        self.cost_weight = cost_weight
        self.rng = random.Random(seed)
        self.path = state_path
        # stats[categorie][brat] = [reușite verificate, eșecuri verificate]
        self.stats: dict[str, dict[str, list[int]]] = {}
        if state_path and state_path.exists():
            self.stats = json.loads(state_path.read_text())

    def _row(self, cat: str, arm: str) -> list[int]:
        return self.stats.setdefault(cat, {}).setdefault(arm, [0, 0])

    def _mean(self, cat: str, arm: str) -> float:
        w, l = self._row(cat, arm)
        return (w + 1) / (w + l + 2)

    def _top3(self, cat: str) -> list[str]:
        return sorted(self.reg.runnable(), key=lambda m: -self._mean(cat, m))[:3]

    def _members(self, cat: str, arm: str) -> list[str]:
        return self._top3(cat) if arm == VOTE else [arm]

    def choose(self, cat: str) -> str:
        arms = self.reg.runnable() + ([VOTE] if len(self.reg.runnable()) >= 3 else [])
        def score(arm: str) -> float:
            w, l = self._row(cat, arm)
            cost = sum(self.reg.cards[m].cost for m in self._members(cat, arm))
            return self.rng.betavariate(w + 1, l + 1) - self.cost_weight * cost
        return max(arms, key=score)

    def run(self, cat: str, prompt: str,
            verifier: Callable[[str], bool]) -> tuple[str, str, bool, float]:
        arm = self.choose(cat)
        members = self._members(cat, arm)
        outs = [self.reg.fns[m](prompt) for m in members]
        out = Counter(outs).most_common(1)[0][0]
        ok = verifier(out)  # singura sursă de adevăr: verificatorul, nu opinia modelelor
        self._row(cat, arm)[0 if ok else 1] += 1
        if self.path:
            self.path.write_text(json.dumps(self.stats))
        return arm, out, ok, sum(self.reg.cards[m].cost for m in members)

    def best(self, cat: str) -> list[tuple[str, float, int]]:
        rows = self.stats.get(cat, {})
        return sorted(((a, self._mean(cat, a), sum(r)) for a, r in rows.items()),
                      key=lambda x: -x[1])
