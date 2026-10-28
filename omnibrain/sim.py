"""Simulare: 30 modele false cu abilități ascunse pe 3 categorii și costuri diferite.
Compară Arena cu alegerea aleatoare și cu 'mereu cel mai scump'."""
import random
from arena import Arena
from registry import ModelCard, Registry

CATS = ["cod", "mate", "text"]
rng = random.Random(7)


def make_model(skill: dict[str, float], seed: int):
    r = random.Random(seed)
    return lambda prompt: prompt if r.random() < skill[prompt.split(":")[0]] else "gresit"


def build() -> Registry:
    reg = Registry()
    for i in range(30):
        skill = {c: rng.random() ** 2 for c in CATS}  # majoritatea sunt slabi
        # modelele mari sunt mai scumpe dar nu neapărat mai bune
        reg.add(ModelCard(f"m{i}", "fake", "mit", cost=rng.choice([1, 1, 2, 5, 10])),
                make_model(skill, i))
    return reg


def evaluate(pick, n=1500, tail=500):
    ok_tail = cost_tail = 0
    for t in range(n):
        cat = CATS[t % 3]
        prompt = f"{cat}:{t}"
        arm, out, ok, cost = pick(cat, prompt)
        if t >= n - tail:
            ok_tail += ok
            cost_tail += cost
    return ok_tail / tail, cost_tail / tail


reg = build()
arena = Arena(reg, cost_weight=0.05, seed=1)
verify = lambda p: (lambda out: out == p)

a = evaluate(lambda c, p: arena.run(c, p, verify(p)))

def rand_pick(c, p):
    m = rng.choice(reg.runnable())
    out = reg.fns[m](p)
    return m, out, out == p, reg.cards[m].cost

def priciest(c, p):
    m = max(reg.runnable(), key=lambda m: reg.cards[m].cost)
    out = reg.fns[m](p)
    return m, out, out == p, reg.cards[m].cost

print(f"{'metodă':<14}{'succes':>8}{'cost/sarcină':>14}")
for name, r in [("Arena", a), ("aleator", evaluate(rand_pick)), ("cel mai scump", evaluate(priciest))]:
    print(f"{name:<14}{r[0]:>8.0%}{r[1]:>14.2f}")
for c in CATS:
    print(c, "->", [(m, round(s, 2), n) for m, s, n in arena.best(c)[:2]])
