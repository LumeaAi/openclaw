import json
import unittest
import urllib.error
import urllib.request

from arena import Arena
from federation import discover, join
from node import Node
from registry import Registry

SECRET = b"federatie-test"
CATS = ["cod", "mate"]


def skilled(good_cat):
    # corect doar pe categoria lui; altfel răspunde sigur, dar greșit
    return lambda p: p if p.split(":")[0] == good_cat else "gresit"


class FederationTest(unittest.TestCase):
    def setUp(self):
        # lanț A-B-C-D-E: clientul cunoaște doar A
        self.nodes: list[Node] = []
        specs = [("A", skilled("cod"), 1), ("B", skilled("mate"), 1),
                 ("C", lambda p: "gresit", 1),            # nod rău-intenționat
                 ("D", skilled("cod"), 5),                # bun dar scump
                 ("E", skilled("mate"), 5)]
        for name, h, cost in reversed(specs):
            peers = [self.nodes[-1].url] if self.nodes else []
            self.nodes.append(Node(name, h, SECRET, cost=cost, peers=peers).start())
        self.nodes.reverse()

    def tearDown(self):
        for n in self.nodes:
            n.stop()

    def test_discovery_finds_whole_chain_from_one_seed(self):
        found = discover([self.nodes[0].url])
        self.assertEqual({c["name"] for c in found.values()}, set("ABCDE"))

    def test_arena_learns_which_peer_to_trust(self):
        reg = Registry()
        join(reg, discover([self.nodes[0].url]), SECRET)
        arena = Arena(reg, seed=2)
        for t in range(300):
            cat = CATS[t % 2]
            p = f"{cat}:{t}"
            arena.run(cat, p, lambda out, p=p: out == p)
        name = lambda cat: reg.cards[arena.best(cat)[0][0]].meta.get("name")
        self.assertEqual(name("cod"), "A")    # bun și ieftin bate D (bun, dar scump)
        self.assertEqual(name("mate"), "B")
        evil = next(u for u, c in reg.cards.items() if c.meta["name"] == "C")
        self.assertEqual(arena.stats["cod"].get(evil, [0, 0])[0], 0)

    def test_unsigned_request_rejected(self):
        body = json.dumps({"prompt": "cod:1"}).encode()
        req = urllib.request.Request(self.nodes[0].url + "/run", body, {"X-Sig": "fals"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 403)

    def test_call_budget_enforced(self):
        n = Node("X", lambda p: p, SECRET, max_calls=2).start()
        try:
            from federation import remote_fn
            f = remote_fn(n.url, SECRET)
            self.assertEqual([f("a"), f("b"), f("c")], ["a", "b", ""])
        finally:
            n.stop()


if __name__ == "__main__":
    unittest.main()
