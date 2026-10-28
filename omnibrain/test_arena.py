import unittest

from arena import Arena
from registry import ModelCard, Registry, FREE_LICENSES


def reg_with(skills: dict[str, float]) -> Registry:
    import random
    reg = Registry()
    for name, p in skills.items():
        r = random.Random(name)
        reg.add(ModelCard(name, "fake", "mit", cost=1),
                lambda prompt, p=p, r=r: prompt if r.random() < p else "gresit")
    return reg


class ArenaTest(unittest.TestCase):
    def test_learns_the_best_model(self):
        reg = reg_with({"bun": 0.95, "mediu": 0.5, "slab": 0.1, "slab2": 0.1})
        arena = Arena(reg, seed=3)
        for t in range(400):
            arena.run("cod", f"q{t}", lambda out, t=t: out == f"q{t}")
        self.assertEqual(arena.best("cod")[0][0], "bun")

    def test_trust_comes_only_from_verifier(self):
        reg = reg_with({"a": 1.0, "b": 1.0, "c": 1.0})
        arena = Arena(reg, seed=1)
        for t in range(50):
            arena.run("x", f"q{t}", lambda out: False)  # verificatorul respinge tot
        self.assertTrue(all(w == 0 for w, _ in arena.stats["x"].values()))

    def test_license_flag(self):
        self.assertTrue(ModelCard("m", "hf", "apache-2.0").free)
        self.assertFalse(ModelCard("m", "hf", "unknown").free)
        self.assertIn("mit", FREE_LICENSES)


if __name__ == "__main__":
    unittest.main()
