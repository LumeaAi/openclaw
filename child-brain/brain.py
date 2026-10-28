"""Creier-copil: filtre + neuroni (agenți/modele) + încredere învățată din feedback.

Fără dependențe externe. Un LLM real se leagă prin `ModelNeuron(llm=callable)`.
"""
from __future__ import annotations

import ast
import json
import operator
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

STOPWORDS = {"care", "este", "e", "sunt", "ce", "cine", "cat", "cum", "a", "al", "ale",
             "de", "la", "in", "pe", "un", "o", "si", "cu", "the", "is", "what", "of"}


def tokens(text: str) -> set[str]:
    plain = unicodedata.normalize("NFD", text.lower())
    plain = "".join(c for c in plain if unicodedata.category(c) != "Mn")
    return {w for w in re.findall(r"\w+", plain) if w not in STOPWORDS}


@dataclass
class Signal:
    text: str
    trace: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    confidence: float  # cât de sigur e neuronul, independent de încrederea câștigată
    neuron: str


# ---------- Filtre senzoriale ----------

class Filter:
    name = "filter"

    def process(self, s: Signal) -> Optional[Signal]:
        raise NotImplementedError


class NoiseFilter(Filter):
    name = "zgomot"

    def process(self, s: Signal) -> Optional[Signal]:
        s.text = re.sub(r"\s+", " ", s.text).strip()
        return s if s.text else None


class SecretFilter(Filter):
    """Nu lăsa chei/parole să ajungă în memorie sau la un model extern."""
    name = "secrete"
    PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")

    def process(self, s: Signal) -> Optional[Signal]:
        cleaned = self.PATTERN.sub("[SECRET]", s.text)
        if cleaned != s.text:
            s.trace.append("secret ascuns")
        s.text = cleaned
        return s


# ---------- Neuroni ----------

class Neuron:
    name = "neuron"

    def __init__(self) -> None:
        self.trust = 0.5  # se schimbă prin feedback

    def fire(self, s: Signal) -> Optional[Answer]:
        raise NotImplementedError


class MemoryNeuron(Neuron):
    name = "memorie"
    MIN_OVERLAP = 0.6

    def __init__(self) -> None:
        super().__init__()
        self.facts: dict[str, str] = {}

    def teach(self, key: str, value: str) -> None:
        self.facts[key.strip()] = value.strip()

    def fire(self, s: Signal) -> Optional[Answer]:
        q = tokens(s.text)
        best, best_score = None, 0.0
        for key, value in self.facts.items():
            k = tokens(key)
            if not k or not q:
                continue
            score = len(q & k) / len(q | k)
            if score > best_score:
                best, best_score = value, score
        if best is not None and best_score >= self.MIN_OVERLAP:
            return Answer(best, best_score, self.name)
        return None


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 100:
            raise ValueError("exponent prea mare")
        return _OPS[type(node.op)](a, b)
    raise ValueError("expresie nepermisă")


class MathNeuron(Neuron):
    name = "matematica"
    EXPR = re.compile(r"[\d(][\d\s+\-*/().%]*[\d)]")

    def fire(self, s: Signal) -> Optional[Answer]:
        for m in self.EXPR.finditer(s.text):
            expr = m.group().strip()
            if not re.search(r"[+\-*/%]", expr):
                continue
            try:
                result = _eval(ast.parse(expr.replace("%", "%"), mode="eval").body)
            except (ValueError, SyntaxError, ZeroDivisionError):
                continue
            shown = int(result) if float(result).is_integer() else round(result, 6)
            return Answer(str(shown), 0.95, self.name)  # verificat prin calcul
        return None


class ModelNeuron(Neuron):
    """Învelește orice LLM. Încrederea de bază e mică: nu e verificat."""
    name = "model"

    def __init__(self, llm: Optional[Callable[[str], str]] = None) -> None:
        super().__init__()
        self.llm = llm

    def fire(self, s: Signal) -> Optional[Answer]:
        if self.llm is None:
            return None
        return Answer(self.llm(s.text), 0.4, self.name)


# ---------- Creierul ----------

class Brain:
    # Sub prag, neuronul tace. Modelul (0.4 * 0.5) începe deasupra lui și e redus
    # la tăcere după câteva greșeli semnalate.
    THRESHOLD = 0.15
    LEARNING_RATE = 0.2

    def __init__(self, state_path: Optional[Path] = None,
                 llm: Optional[Callable[[str], str]] = None) -> None:
        self.filters: list[Filter] = [NoiseFilter(), SecretFilter()]
        self.memory = MemoryNeuron()
        self.neurons: list[Neuron] = [self.memory, MathNeuron(), ModelNeuron(llm)]
        self.state_path = state_path
        self._last: Optional[tuple[Signal, list[Answer], Answer]] = None
        self._load()

    # percepție -> răspuns
    def ask(self, text: str) -> str:
        s: Optional[Signal] = Signal(text)
        for f in self.filters:
            s = f.process(s) if s else None
        if s is None:
            return "(nu am primit nimic)"
        candidates = [a for n in self.neurons if (a := n.fire(s))]
        scored = [(a.confidence * self._neuron(a.neuron).trust, a) for a in candidates]
        scored = [(sc, a) for sc, a in scored if sc >= self.THRESHOLD]
        if not scored:
            self._last = None
            return "Nu știu încă. Mă înveți? (/invata întrebare = răspuns)"
        best = max(scored, key=lambda x: x[0])[1]
        self._last = (s, candidates, best)
        return f"{best.text}   [{best.neuron}]"

    # învățare
    def teach(self, key: str, value: str) -> None:
        key = SecretFilter().process(Signal(key)).text
        value = SecretFilter().process(Signal(value)).text
        self.memory.teach(key, value)
        self._save()

    def feedback(self, good: bool, correction: Optional[str] = None) -> str:
        if self._last is None:
            return "Nu am un răspuns recent de evaluat."
        s, candidates, winner = self._last
        target = 1.0 if good else 0.0
        n = self._neuron(winner.neuron)
        n.trust += self.LEARNING_RATE * (target - n.trust)
        # neuronii care au spus același lucru împart soarta câștigătorului
        for a in candidates:
            if a is not winner and a.text == winner.text:
                m = self._neuron(a.neuron)
                m.trust += self.LEARNING_RATE * (target - m.trust)
        if not good and correction:
            self.teach(s.text, correction)
        self._save()
        return f"Am învățat. Încredere {n.name}: {n.trust:.2f}"

    def status(self) -> str:
        lines = [f"{n.name:<11} încredere {n.trust:.2f}" for n in self.neurons]
        lines.append(f"fapte învățate: {len(self.memory.facts)}")
        return "\n".join(lines)

    def _neuron(self, name: str) -> Neuron:
        return next(n for n in self.neurons if n.name == name)

    # persistență
    def _save(self) -> None:
        if not self.state_path:
            return
        data = {"facts": self.memory.facts, "trust": {n.name: n.trust for n in self.neurons}}
        self.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        self.memory.facts = data.get("facts", {})
        for n in self.neurons:
            n.trust = data.get("trust", {}).get(n.name, n.trust)
