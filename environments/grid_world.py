"""
Grid World : 4×4, piège haut-droite (-1), objectif bas-droite (+1).
Contrats MDPEnv et ModelFreeEnv pour tester tous les algos.

État = ligne * nb_colonnes + colonne.
Les bords reflètent (l'agent reste sur place, reward 0).
"""

from typing import List

from environments.base import MDPEnv, ModelFreeEnv

# Actions (indices imposés par les contrats : des entiers de 0 à 3).
HAUT = 0
BAS = 1
GAUCHE = 2
DROITE = 3

NOMS_ACTIONS = {HAUT: "haut", BAS: "bas", GAUCHE: "gauche", DROITE: "droite"}


class GridWorld(MDPEnv, ModelFreeEnv):
    """Grille rectangulaire avec un objectif (+1) et un piège (-1)."""

    def __init__(self, nb_lignes: int = 4, nb_colonnes: int = 4) -> None:
        if nb_lignes < 2 or nb_colonnes < 2:
            raise ValueError(
                "Il faut au moins 2 lignes et 2 colonnes pour que le départ, "
                "le piège et l'objectif soient des cases distinctes."
            )
        self.nb_lignes = nb_lignes
        self.nb_colonnes = nb_colonnes

        # Cases particulières (indices d'état aplatis).
        self.depart = 0                                    # coin haut-gauche
        self.piege = nb_colonnes - 1                       # coin haut-droit
        self.objectif = nb_lignes * nb_colonnes - 1        # coin bas-droit

        # Les trois rewards possibles de l'environnement, indexés pour le
        # contrat MDPEnv (num_rewards / reward(i) / p(..., r_index)).
        self._recompenses = [-1.0, 0.0, 1.0]

        # Partie ModelFreeEnv : position courante et score cumulé.
        self.reset()

    # ------------------------------------------------------------------ #
    # Dynamique commune aux deux contrats                                  #
    # ------------------------------------------------------------------ #

    def _est_terminal(self, s: int) -> bool:
        return s == self.piege or s == self.objectif

    def _destination(self, s: int, a: int) -> int:
        """
        État d'arrivée quand on joue l'action a depuis l'état s (déterministe).
        Un déplacement qui sortirait de la grille laisse l'agent sur place.
        """
        ligne, colonne = divmod(s, self.nb_colonnes)
        if a == HAUT:
            ligne = max(ligne - 1, 0)
        elif a == BAS:
            ligne = min(ligne + 1, self.nb_lignes - 1)
        elif a == GAUCHE:
            colonne = max(colonne - 1, 0)
        elif a == DROITE:
            colonne = min(colonne + 1, self.nb_colonnes - 1)
        else:
            raise ValueError(f"Action inconnue : {a}")
        return ligne * self.nb_colonnes + colonne

    def _recompense_arrivee(self, s_p: int) -> float:
        """Reward touché en entrant dans l'état s_p."""
        if s_p == self.piege:
            return -1.0
        if s_p == self.objectif:
            return 1.0
        return 0.0

    # ------------------------------------------------------------------ #
    # Contrat MDPEnv (Dynamic Programming)                                 #
    # ------------------------------------------------------------------ #

    def num_states(self) -> int:
        return self.nb_lignes * self.nb_colonnes

    def num_actions(self) -> int:
        return 4

    def num_rewards(self) -> int:
        return len(self._recompenses)

    def reward(self, i: int) -> float:
        return self._recompenses[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        """p(s', r | s, a). Déterministe : une seule (s', r) possible."""
        if self._est_terminal(s):
            return 0.0
        destination = self._destination(s, a)
        if s_p != destination:
            return 0.0
        if self._recompenses[r_index] != self._recompense_arrivee(destination):
            return 0.0
        return 1.0

    # Contrat ModelFreeEnv (Monte Carlo, TD, Planning)

    def reset(self) -> None:
        self._etat = self.depart
        self._score = 0.0

    def step(self, action: int) -> None:
        if self.is_game_over():
            raise ValueError("Épisode terminé : appeler reset() avant step().")
        destination = self._destination(self._etat, action)
        self._score += self._recompense_arrivee(destination)
        self._etat = destination

    def is_game_over(self) -> bool:
        return self._est_terminal(self._etat)

    def current_state(self) -> int:
        return self._etat

    def available_actions(self) -> List[int]:
        if self.is_game_over():
            return []
        return [HAUT, BAS, GAUCHE, DROITE]

    def score(self) -> float:
        return self._score

    def maximum_states_count(self) -> int:
        return self.num_states()

    def maximum_actions_count(self) -> int:
        return self.num_actions()

    def pretty_print(self) -> None:
        for ligne in range(self.nb_lignes):
            cases = []
            for colonne in range(self.nb_colonnes):
                s = ligne * self.nb_colonnes + colonne
                if s == self._etat:
                    cases.append("A")
                elif s == self.objectif:
                    cases.append("G")
                elif s == self.piege:
                    cases.append("X")
                else:
                    cases.append(".")
            print(" ".join(cases))
        print(f"score = {self._score}")


# Mode humain : python -m environments.grid_world
# Touches sans Entrée (termios/tty en cbreak) + écran redessiné par ANSI.

if __name__ == "__main__":
    import os
    import select
    import sys
    import termios
    import tty

    ZQSD = {"z": HAUT, "s": BAS, "q": GAUCHE, "d": DROITE}
    FLECHES = {b"[A": HAUT, b"[B": BAS, b"[D": GAUCHE, b"[C": DROITE}

    def lire_touche() -> object:
        # os.read(), pas sys.stdin.read() : stdin bufferise les 3 octets ESC[X
        fd = sys.stdin.fileno()
        reglages = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            c = os.read(fd, 1)
            if c in (b"\x03", b"\x04"):
                return "quitter"
            if c == b"\x1b":
                if select.select([fd], [], [], 0.05)[0]:
                    suite = os.read(fd, 2)
                    if suite in FLECHES:
                        return FLECHES[suite]
                return "quitter"
            car = c.decode(errors="ignore").lower()
            if car in ZQSD:
                return ZQSD[car]
            return car
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, reglages)

    def dessiner(env: GridWorld, journal: list) -> None:
        print("\033[H\033[2J", end="")
        print("GRID WORLD — flèches ou ZQSD pour bouger, Échap pour quitter")
        print("But : atteindre G (+1) en évitant X (-1)\n")
        env.pretty_print()

        ligne, colonne = divmod(env.current_state(), env.nb_colonnes)
        print(f"\ns (current_state)   = {env.current_state()}  "
              f"(ligne={ligne}, colonne={colonne})")
        print(f"available_actions() = {env.available_actions()} "
              f"({[NOMS_ACTIONS[a] for a in env.available_actions()]})")
        print(f"is_game_over()      = {env.is_game_over()}")
        print(f"score()             = {env.score()}")

        print("\n--- journal (s, a, r=Δscore, s') ---------------------")
        if not journal:
            print("(aucun coup joué)")
        for entree in journal[-8:]:
            print(
                f"  s={entree['s']:>2}  a={NOMS_ACTIONS[entree['a']]:<7}"
                f"  r={entree['r']:+.1f}  ->  s'={entree['s_p']:>2}"
                f"  (score={entree['score']:+.1f})"
            )

    env = GridWorld()
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
            if isinstance(touche, int):
                s_avant = env.current_state()
                prev_score = env.score()
                env.step(touche)
                r = env.score() - prev_score
                journal.append({
                    "s": s_avant, "a": touche, "s_p": env.current_state(),
                    "r": r, "score": env.score(),
                })
    finally:
        print("\033[?25h", end="")
