"""
Two-Round Rock Paper Scissors : l'adversaire rejoue notre coup au round 2.
Contrats MDPEnv et ModelFreeEnv.

État encode le round et le coup du round 1 : 0 (round 1), 1-3 (round 2),
4 (terminal). Score final ∈ {-2, -1, 0, 1, 2}.
"""

import random
from typing import List, Optional

from environments.base import MDPEnv, ModelFreeEnv

# Actions (les mêmes aux deux rounds).
PIERRE = 0
FEUILLE = 1
CISEAUX = 2

NOMS_ACTIONS = {PIERRE: "pierre", FEUILLE: "feuille", CISEAUX: "ciseaux"}


def _compare(a: int, b: int) -> int:
    # (a - b) % 3 == 1 ⟺ a bat b (formule modulaire plus lisible qu'une table)
    if a == b:
        return 0
    return 1 if (a - b) % 3 == 1 else -1


class TwoRoundRPS(MDPEnv, ModelFreeEnv):
    """Pierre-Feuille-Ciseaux en 2 rounds, l'adversaire rejouant l'agent."""

    # États particuliers (indices, cf. encodage détaillé dans le docstring).
    DEPART = 0
    TERMINAL = 4

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        # rng injectable pour rendre les tirages de l'adversaire
        # reproductibles pendant les tests/expérimentations.
        self._rng = rng if rng is not None else random.Random()
        self._recompenses = [-1.0, 0.0, 1.0]
        self.reset()


    def _est_terminal(self, s: int) -> bool:
        return s == self.TERMINAL

    def _est_round_2(self, s: int) -> bool:
        return s in (1, 2, 3)

    # Contrat MDPEnv

    def num_states(self) -> int:
        return 5

    def num_actions(self) -> int:
        return 3

    def num_rewards(self) -> int:
        return len(self._recompenses)

    def reward(self, i: int) -> float:
        return self._recompenses[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        """p(s', r | s, a). Round 1 : stochastique (1/3 chaque issue).
        Round 2 : déterministe (adversaire forcé rejouer)."""
        if self._est_terminal(s):
            return 0.0

        r = self._recompenses[r_index]

        if s == self.DEPART:
            if s_p != 1 + a:
                return 0.0
            return 1.0 / 3.0

        # s ∈ {1, 2, 3} : round 2.
        if s_p != self.TERMINAL:
            return 0.0
        coup_round1 = s - 1
        resultat = _compare(a, coup_round1)
        return 1.0 if r == resultat else 0.0

    # Contrat ModelFreeEnv

    def reset(self) -> None:
        self._round = 1
        self._coup_round1 = None
        self._dernier_coup_adverse = None
        self._score = 0.0

    def step(self, action: int) -> None:
        if self.is_game_over():
            raise ValueError("Épisode terminé : appeler reset() avant step().")

        if self._round == 1:
            coup_adverse = self._rng.randrange(3)
            self._coup_round1 = action
        else:  # round 2 : l'adversaire est forcé de rejouer le round 1
            coup_adverse = self._coup_round1

        self._score += _compare(action, coup_adverse)
        self._dernier_coup_adverse = coup_adverse
        self._round += 1

    def is_game_over(self) -> bool:
        return self._round == 3

    def current_state(self) -> int:
        if self._round == 1:
            return self.DEPART
        if self._round == 2:
            return 1 + self._coup_round1
        return self.TERMINAL

    def available_actions(self) -> List[int]:
        if self.is_game_over():
            return []
        return [PIERRE, FEUILLE, CISEAUX]

    def score(self) -> float:
        return self._score

    def maximum_states_count(self) -> int:
        return self.num_states()

    def maximum_actions_count(self) -> int:
        return self.num_actions()

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def pretty_print(self) -> None:
        """Affiche le round courant, les coups joués et le score cumulé."""
        if self._round <= 2:
            print(f"Round {min(self._round, 2)}/2")
        else:
            print("Partie terminée")
        if self._coup_round1 is not None:
            print(f"  coup de l'agent au round 1 : {NOMS_ACTIONS[self._coup_round1]}")
        if self._dernier_coup_adverse is not None:
            print(f"  dernier coup de l'adversaire : {NOMS_ACTIONS[self._dernier_coup_adverse]}")
        print(f"  score = {self._score:+.1f}")


# Mode humain : python -m environments.rock_paper_scissors
# Même principe que grid_world.py.

if __name__ == "__main__":
    import os
    import select
    import sys
    import termios
    import tty

    TOUCHES = {"p": PIERRE, "f": FEUILLE, "c": CISEAUX}

    def lire_touche() -> object:
        fd = sys.stdin.fileno()
        reglages = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            c = os.read(fd, 1)
            if c in (b"\x03", b"\x04", b"\x1b"):   # Ctrl-C / Ctrl-D / Échap
                return "quitter"
            car = c.decode(errors="ignore").lower()
            return TOUCHES.get(car, car)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, reglages)

    def dessiner(env: TwoRoundRPS, journal: list) -> None:
        print("\033[H\033[2J", end="")
        print("TWO-ROUND ROCK PAPER SCISSORS — p=pierre f=feuille c=ciseaux, Échap pour quitter")
        print("Round 2 : l'adversaire est forcé de rejouer TON coup du round 1.\n")
        env.pretty_print()

        print(f"\ns (current_state)   = {env.current_state()}")
        print(f"available_actions() = {env.available_actions()} "
              f"({[NOMS_ACTIONS[a] for a in env.available_actions()]})")
        print(f"is_game_over()      = {env.is_game_over()}")
        print(f"score()             = {env.score():+.1f}")

        print("\n--- journal (s, a, adversaire, r=Δscore, s') -----------")
        if not journal:
            print("(aucun coup joué)")
        for entree in journal:
            print(
                f"  s={entree['s']}  a={NOMS_ACTIONS[entree['a']]:<7}"
                f"  adversaire={NOMS_ACTIONS[entree['adversaire']]:<7}"
                f"  r={entree['r']:+.1f}  ->  s'={entree['s_p']}"
                f"  (score={entree['score']:+.1f})"
            )

    env = TwoRoundRPS()
    journal: list = []
    print("\033[?25l", end="")
    try:
        while True:
            dessiner(env, journal)
            if env.is_game_over():
                if env.score() > 0:
                    resultat = "Victoire !"
                elif env.score() < 0:
                    resultat = "Défaite..."
                else:
                    resultat = "Match nul."
                print(f"\n{resultat}")
                print("[R] rejouer — [autre touche] quitter")
                if lire_touche() != "r":
                    break
                env.reset()
                journal.clear()
                continue
            touche = lire_touche()
            if touche == "quitter":
                break
            if isinstance(touche, int):
                s_avant = env.current_state()
                prev_score = env.score()
                env.step(touche)
                r = env.score() - prev_score
                journal.append({
                    "s": s_avant, "a": touche,
                    "adversaire": env._dernier_coup_adverse,
                    "r": r, "s_p": env.current_state(), "score": env.score(),
                })
    finally:
        print("\033[?25h", end="")
