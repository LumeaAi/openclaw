"""Conversație cu creierul-copil. Rulează: python3 chat.py"""
from pathlib import Path

from brain import Brain

HELP = """/invata întrebare = răspuns   îl înveți un fapt
/bun                          ultimul răspuns a fost corect
/gresit [răspuns corect]      ultimul răspuns a fost greșit (opțional îl corectezi)
/stare                        vezi încrederea neuronilor
/iesire"""


def main() -> None:
    brain = Brain(state_path=Path(__file__).with_name("state.json"))
    print("Sunt un creier-copil. Nu știu aproape nimic. Învață-mă!\n" + HELP)
    while True:
        try:
            line = input("\ntu> ").strip()
        except EOFError:
            break
        if line == "/iesire":
            break
        elif line.startswith("/invata") and "=" in line:
            key, value = line[len("/invata"):].split("=", 1)
            brain.teach(key, value)
            print("Am reținut.")
        elif line == "/bun":
            print(brain.feedback(True))
        elif line.startswith("/gresit"):
            print(brain.feedback(False, line[len("/gresit"):].strip() or None))
        elif line == "/stare":
            print(brain.status())
        elif line.startswith("/"):
            print(HELP)
        else:
            print(brain.ask(line))


if __name__ == "__main__":
    main()
