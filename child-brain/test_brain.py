import unittest

from brain import Brain


class ChildBrainTest(unittest.TestCase):
    def test_starts_ignorant(self):
        self.assertIn("Nu știu", Brain().ask("Care este capitala Franței?"))

    def test_learns_and_recalls(self):
        b = Brain()
        b.teach("capitala Franței", "Paris")
        self.assertTrue(b.ask("Care este capitala Franței?").startswith("Paris"))

    def test_math_is_computed_not_guessed(self):
        self.assertTrue(Brain().ask("cât face 12*(3+4)?").startswith("84"))

    def test_math_rejects_code(self):
        self.assertIn("Nu știu", Brain().ask("__import__('os').system('ls')"))

    def test_feedback_moves_trust(self):
        b = Brain(llm=lambda q: "Londra")
        b.ask("Care este capitala Franței?")
        before = b.neurons[2].trust
        b.feedback(False)
        self.assertLess(b.neurons[2].trust, before)

    def test_correction_teaches(self):
        b = Brain(llm=lambda q: "Londra")
        b.ask("capitala Franței")
        b.feedback(False, "Paris")
        self.assertTrue(b.ask("capitala Franței").startswith("Paris"))

    def test_secret_is_redacted_before_memory(self):
        b = Brain()
        b.teach("cheia mea", "sk-abcdefghijklmnopqrstuv")
        self.assertNotIn("sk-abc", str(b.memory.facts))


if __name__ == "__main__":
    unittest.main()
