"""
Monty Hall niveau 2 : 5 portes, 4 décisions, reward uniquement à la fin.
État encode l'historique des décisions 2 et 3 pour être markovien.
Contrats MDPEnv (proba exacte par énumération) et ModelFreeEnv.
"""

import random
from fractions import Fraction
from typing import List, Optional

from environments.base import MDPEnv, ModelFreeEnv

NB_PORTES = 5

# Actions des décisions 2, 3 et 4. À la décision 1, l'action est
# directement l'indice de la porte choisie (0 à 4).
GARDER = 0
CHANGER = 1

NOMS_DECISIONS = {GARDER: "garder", CHANGER: "changer"}


def _proba_tenue_gagnante(decision_2: int, decision_3: int) -> Fraction:
    """Proba exacte tenue gagnante à la décision 4. Énumération exhaustive."""
    # Départ : l'agent tient la porte 0, la gagnante est uniforme.
    configs = {
        (frozenset(range(NB_PORTES)), 0, gagnante): Fraction(1, NB_PORTES)
        for gagnante in range(NB_PORTES)
    }

    # Le jeu alterne : [décision] puis retrait, trois fois. `None` marque le
    # premier tour (la décision 1 est déjà appliquée : tenue = porte 0).
    for decision in (None, decision_2, decision_3):
        # a) la décision de l'agent (changer = une autre porte, équiprobable)
        if decision == CHANGER:
            apres = {}
            for (portes, tenue, gagnante), proba in configs.items():
                autres = [pt for pt in portes if pt != tenue]
                for cible in autres:
                    cle = (portes, cible, gagnante)
                    apres[cle] = apres.get(cle, Fraction(0)) + proba / len(autres)
            configs = apres

        # b) le présentateur retire une porte non tenue et non gagnante,
        #    équiprobable parmi les candidates
        apres = {}
        for (portes, tenue, gagnante), proba in configs.items():
            candidates = [pt for pt in portes if pt != tenue and pt != gagnante]
            for retrait in candidates:
                cle = (portes - {retrait}, tenue, gagnante)
                apres[cle] = apres.get(cle, Fraction(0)) + proba / len(candidates)
        configs = apres

    return sum(
        proba
        for (portes, tenue, gagnante), proba in configs.items()
        if tenue == gagnante
    )


class MontyHall2(MDPEnv, ModelFreeEnv):
    """Monty Hall à 5 portes et 4 décisions (état = historique des choix)."""

    DEPART = 0
    TERMINAL = 8

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._recompenses = [0.0, 1.0]
        # P(porte tenue gagnante à la décision 4 | décisions 2 et 3) —
        # exact, calculé une fois pour toutes pour le modèle MDPEnv.
        self._proba_gagne_si_garde = {
            (d2, d3): _proba_tenue_gagnante(d2, d3)
            for d2 in (GARDER, CHANGER)
            for d3 in (GARDER, CHANGER)
        }
        self.reset()

    # Contrat MDPEnv

    def num_states(self) -> int:
        return 9

    def num_actions(self) -> int:
        return NB_PORTES  # 5, pour couvrir le choix de porte de la décision 1

    def num_rewards(self) -> int:
        return len(self._recompenses)

    def reward(self, i: int) -> float:
        return self._recompenses[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        """p(s', r | s, a). Actions fantômes (a ≥ 2) hors départ : p = 0."""
        if s == self.TERMINAL:
            return 0.0
        r = self._recompenses[r_index]

        if s == self.DEPART:
            # Décision 1 : les 5 portes sont symétriques, tout choix mène
            # à l'état 1 avec certitude et reward 0.
            return 1.0 if (s_p == 1 and r == 0.0) else 0.0

        if a not in (GARDER, CHANGER):
            return 0.0

        if s == 1:  # décision 2
            return 1.0 if (s_p == 2 + a and r == 0.0) else 0.0

        if s in (2, 3):  # décision 3, d2 = s - 2
            d2 = s - 2
            return 1.0 if (s_p == 4 + 2 * d2 + a and r == 0.0) else 0.0

        # s ∈ {4..7} : décision finale, ouverture de la porte.
        if s_p != self.TERMINAL:
            return 0.0
        d2, d3 = divmod(s - 4, 2)
        q = self._proba_gagne_si_garde[(d2, d3)]
        proba_gagner = q if a == GARDER else 1 - q
        return float(proba_gagner if r == 1.0 else 1 - proba_gagner)

    # Contrat ModelFreeEnv

    def reset(self) -> None:
        self._gagnante = self._rng.randrange(NB_PORTES)
        self._portes = set(range(NB_PORTES))   # portes encore en jeu
        self._tenue: Optional[int] = None      # porte tenue par l'agent
        self._historique: List[int] = []       # décisions 2 et 3
        self._decision = 1                     # prochaine décision (1..4)
        self._score = 0.0

    def _retirer_une_porte(self) -> None:
        candidates = sorted(self._portes - {self._tenue, self._gagnante})
        self._portes.remove(self._rng.choice(candidates))

    def step(self, action: int) -> None:
        if self.is_game_over():
            raise ValueError("Épisode terminé : appeler reset() avant step().")

        if self._decision == 1:
            if action not in range(NB_PORTES):
                raise ValueError(f"Décision 1 : choisir une porte 0..4, reçu {action}")
            self._tenue = action
            self._retirer_une_porte()
        elif self._decision in (2, 3):
            if action not in (GARDER, CHANGER):
                raise ValueError(f"Décisions 2-4 : garder(0)/changer(1), reçu {action}")
            if action == CHANGER:
                autres = sorted(self._portes - {self._tenue})
                self._tenue = self._rng.choice(autres)
            self._historique.append(action)
            self._retirer_une_porte()
        else:  # décision 4 : ouverture
            if action not in (GARDER, CHANGER):
                raise ValueError(f"Décisions 2-4 : garder(0)/changer(1), reçu {action}")
            if action == CHANGER:
                (self._tenue,) = self._portes - {self._tenue}
            if self._tenue == self._gagnante:
                self._score += 1.0

        self._decision += 1

    def is_game_over(self) -> bool:
        return self._decision > 4

    def current_state(self) -> int:
        if self._decision == 1:
            return self.DEPART
        if self._decision == 2:
            return 1
        if self._decision == 3:
            return 2 + self._historique[0]
        if self._decision == 4:
            return 4 + 2 * self._historique[0] + self._historique[1]
        return self.TERMINAL

    def available_actions(self) -> List[int]:
        if self.is_game_over():
            return []
        if self._decision == 1:
            return list(range(NB_PORTES))
        return [GARDER, CHANGER]

    def score(self) -> float:
        return self._score

    def maximum_states_count(self) -> int:
        return self.num_states()

    def maximum_actions_count(self) -> int:
        return self.num_actions()

    def pretty_print(self) -> None:
        cases = []
        for porte in range(NB_PORTES):
            nom = chr(ord("A") + porte)
            if porte not in self._portes:
                cases.append(" ✗ ")
            elif porte == self._tenue:
                cases.append(f"[{nom}]")
            else:
                cases.append(f" {nom} ")
        print("Portes :" + " ".join(cases))
        if self.is_game_over():
            print(f"Partie terminée — la gagnante était {chr(ord('A') + self._gagnante)}")
        else:
            print(f"Décision {self._decision}/4")
        print(f"score = {self._score}")


# Mode humain : python -m environments.monty_hall_2

if __name__ == "__main__":
    import os
    import select
    import sys
    import termios
    import tty

    def lire_touche() -> str:
        fd = sys.stdin.fileno()
        reglages = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            c = os.read(fd, 1)
            if c in (b"\x03", b"\x04", b"\x1b"):   # Ctrl-C / Ctrl-D / Échap
                return "quitter"
            return c.decode(errors="ignore").lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, reglages)

    def nom_action(s: int, a: int) -> str:
        if s == MontyHall2.DEPART:
            return f"porte {chr(ord('A') + a)}"
        return NOMS_DECISIONS[a]

    def dessiner(env: MontyHall2, journal: list) -> None:
        print("\033[H\033[2J", end="")
        print("MONTY HALL NIVEAU 2 — 5 portes, 4 décisions, Échap pour quitter")
        print("Décision 1 : touches A-E. Décisions 2-4 : G garder, C changer.\n")
        env.pretty_print()

        print(f"\ns (current_state)   = {env.current_state()}")
        print(f"available_actions() = {env.available_actions()}")
        print(f"is_game_over()      = {env.is_game_over()}")
        print(f"score()             = {env.score()}")

        print("\n--- journal (s, a, r=Δscore, s') ---------------------")
        if not journal:
            print("(aucune décision prise)")
        for entree in journal:
            print(
                f"  s={entree['s']}  a={entree['a']:<8}"
                f"  r={entree['r']:+.1f}  ->  s'={entree['s_p']}"
                f"  (score={entree['score']:+.1f})"
            )

    env = MontyHall2()
    journal: list = []
    print("\033[?25l", end="")
    try:
        while True:
            dessiner(env, journal)
            if env.is_game_over():
                print("\n" + ("Gagné !" if env.score() > 0 else "Perdu..."))
                print("[R] rejouer — [autre touche] quitter")
                if lire_touche() != "r":
                    break
                env.reset()
                journal.clear()
                continue
            touche = lire_touche()
            if touche == "quitter":
                break
            # Traduire la touche en action selon la décision en cours.
            action = None
            if env.current_state() == MontyHall2.DEPART:
                if touche in "abcde":
                    action = ord(touche) - ord("a")
            else:
                if touche == "g":
                    action = GARDER
                elif touche == "c":
                    action = CHANGER
            if action is None:
                continue
            s_avant = env.current_state()
            prev_score = env.score()
            env.step(action)
            journal.append({
                "s": s_avant, "a": nom_action(s_avant, action),
                "r": env.score() - prev_score,
                "s_p": env.current_state(), "score": env.score(),
            })
    finally:
        print("\033[?25h", end="")
