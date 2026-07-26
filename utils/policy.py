import random
from typing import List

# Une politique associe a chaque etat une distribution de probabilite sur les actions.
# probs[s][a] = probabilite de choisir l'action a dans l'etat s (la ligne s somme a 1).


class Politique:

    def __init__(self, probs: List[List[float]]) -> None:
        self._probs = probs

    # ---------------- constructeurs ----------------

    @classmethod
    def uniforme(cls, nb_etats: int, nb_actions: int) -> "Politique":
        # chaque action a la meme probabilite dans chaque etat
        p = 1.0 / nb_actions
        return cls([[p] * nb_actions for _ in range(nb_etats)])

    @classmethod
    def deterministe(cls, action_par_etat: List[int], nb_actions: int) -> "Politique":
        # action_par_etat[s] = l'unique action choisie dans l'etat s (probabilite 1.0)
        probs = []
        for a in action_par_etat:
            ligne = [0.0] * nb_actions
            ligne[a] = 1.0
            probs.append(ligne)
        return cls(probs)

    @classmethod
    def gloutonne(cls, q_values: List[List[float]]) -> "Politique":
        # dans chaque etat, l'action de plus grande valeur Q (argmax)
        probs = []
        for valeurs in q_values:
            meilleure = max(valeurs)
            gagnantes = [a for a, v in enumerate(valeurs) if v == meilleure]
            p = 1.0 / len(gagnantes)
            ligne = [0.0] * len(valeurs)
            for a in gagnantes:
                ligne[a] = p
            probs.append(ligne)
        return cls(probs)

    @classmethod
    def epsilon_gloutonne(cls, q_values: List[List[float]], epsilon: float) -> "Politique":
        # exploration epsilon (uniforme sur toutes les actions) + exploitation (1-epsilon) sur l'argmax
        probs = []
        for valeurs in q_values:
            nb_a = len(valeurs)
            ligne = [epsilon / nb_a] * nb_a
            meilleure = max(valeurs)
            gagnantes = [a for a, v in enumerate(valeurs) if v == meilleure]
            bonus = (1.0 - epsilon) / len(gagnantes)
            for a in gagnantes:
                ligne[a] += bonus
            probs.append(ligne)
        return cls(probs)

    # ---------------- lecture ----------------

    def proba(self, etat: int, action: int) -> float:
        return self._probs[etat][action]

    def meilleure_action(self, etat: int) -> int:
        ligne = self._probs[etat]
        meilleure = 0
        for a in range(1, len(ligne)):
            if ligne[a] > ligne[meilleure]:
                meilleure = a
        return meilleure

    def echantillonner(self, etat: int) -> int:
        # tire une action au hasard selon la distribution de l'etat (methode de la roulette)
        tirage = random.random()
        cumul = 0.0
        ligne = self._probs[etat]
        for action, p in enumerate(ligne):
            cumul += p
            if tirage < cumul:
                return action
        return len(ligne) - 1  # filet contre l'imprecision flottante (cumul < 1.0)
