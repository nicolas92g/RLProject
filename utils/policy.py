"""
Représentation unifiée d'une politique.

Une politique associe à chaque état une distribution de probabilité sur les
actions. Ce format unique couvre les deux besoins du projet :

- politique **déterministe** (Dynamic Programming, extraction gloutonne d'une
  fonction de valeur) : une seule action porte la probabilité 1.0 ;
- politique **stochastique** (ε-greedy en Monte Carlo et TD) : la masse de
  probabilité est répartie sur plusieurs actions.

Un seul type pour les deux évite que chaque algorithme invente sa propre
convention, ce qui compliquerait toute comparaison entre algorithmes.

Le stockage est en Python pur (listes + module `random`) pour ne pas imposer
de dépendance à ce niveau. numpy arrivera avec les algorithmes qui en ont
réellement besoin (typiquement la vectorisation de Value Iteration).

Convention : les états et les actions sont des indices entiers, comme dans les
contrats de `environments/base.py`. Une politique est un tableau de forme
(nb_états, nb_actions) où `probs[s][a] = π(a | s)`.
"""

import random
from typing import Dict, List, Optional, Sequence

# Une fonction de valeur d'action Q est stockée comme la politique :
# q_values[s][a] = estimation de Q(s, a).
ActionValues = Sequence[Sequence[float]]


class Politique:
    """Distribution de probabilité sur les actions, pour chaque état."""

    def __init__(self, probabilites: List[List[float]]) -> None:
        """
        `probabilites[s][a]` est la probabilité de choisir l'action a dans
        l'état s. Chaque ligne doit sommer à 1 (à la tolérance flottante près).
        On ne recopie pas la structure : l'appelant en cède la propriété.
        """
        self._probs = probabilites

    # ------------------------------------------------------------------ #
    # Constructeurs                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def uniforme(cls, nb_etats: int, nb_actions: int) -> "Politique":
        """Politique uniforme : chaque action équiprobable dans chaque état."""
        p = 1.0 / nb_actions
        return cls([[p] * nb_actions for _ in range(nb_etats)])

    @classmethod
    def deterministe(
        cls, action_par_etat: Sequence[int], nb_actions: int
    ) -> "Politique":
        """
        Politique déterministe à partir d'une action par état.
        `action_par_etat[s]` est l'action choisie dans l'état s (proba 1.0).
        """
        probs = []
        for a in action_par_etat:
            ligne = [0.0] * nb_actions
            ligne[a] = 1.0
            probs.append(ligne)
        return cls(probs)

    @classmethod
    def gloutonne(cls, q_values: ActionValues) -> "Politique":
        """
        Politique gloutonne (déterministe) extraite d'une fonction de valeur
        d'action : dans chaque état, l'action de plus grande valeur. En cas
        d'égalité, la masse est répartie également entre les actions ex æquo,
        ce qui évite un biais systématique vers le plus petit indice.
        """
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
    def epsilon_gloutonne(
        cls, q_values: ActionValues, epsilon: float
    ) -> "Politique":
        """
        Politique ε-greedy extraite d'une fonction de valeur d'action.

        Avec probabilité ε on explore (action uniforme parmi toutes les
        actions), sinon on exploite (action gloutonne). Concrètement, chaque
        action reçoit ε/|A|, et la (les) action(s) gloutonne(s) reçoivent en
        plus (1-ε) répartie entre elles.
        """
        probs = []
        for valeurs in q_values:
            nb_a = len(valeurs)
            base = epsilon / nb_a
            ligne = [base] * nb_a
            meilleure = max(valeurs)
            gagnantes = [a for a, v in enumerate(valeurs) if v == meilleure]
            bonus = (1.0 - epsilon) / len(gagnantes)
            for a in gagnantes:
                ligne[a] += bonus
            probs.append(ligne)
        return cls(probs)

    # ------------------------------------------------------------------ #
    # Lecture                                                             #
    # ------------------------------------------------------------------ #

    def probabilites(self, etat: int) -> List[float]:
        """Distribution complète sur les actions dans cet état."""
        return self._probs[etat]

    def proba(self, etat: int, action: int) -> float:
        """Probabilité π(action | état)."""
        return self._probs[etat][action]

    def meilleure_action(self, etat: int) -> int:
        """
        Action de plus forte probabilité dans cet état. En cas d'égalité,
        renvoie l'indice le plus petit (déterministe et reproductible) — pour
        un tirage aléatoire parmi les ex æquo, utiliser `echantillonner`.
        """
        ligne = self._probs[etat]
        return max(range(len(ligne)), key=ligne.__getitem__)

    def echantillonner(
        self, etat: int, rng: Optional[random.Random] = None
    ) -> int:
        """
        Tire une action selon la distribution de l'état. Un `random.Random`
        peut être fourni pour la reproductibilité (seed d'expérimentation).
        """
        tirage = rng.random() if rng is not None else random.random()
        cumul = 0.0
        ligne = self._probs[etat]
        for action, p in enumerate(ligne):
            cumul += p
            if tirage < cumul:
                return action
        # Filet contre l'imprécision flottante (cumul légèrement < 1.0) :
        # renvoyer la dernière action de probabilité non nulle.
        for action in range(len(ligne) - 1, -1, -1):
            if ligne[action] > 0.0:
                return action
        raise ValueError(f"Distribution vide pour l'état {etat}")

    def est_deterministe(self, etat: int, tol: float = 1e-9) -> bool:
        """Vrai si une action porte (presque) toute la probabilité."""
        return any(p > 1.0 - tol for p in self._probs[etat])

    # ------------------------------------------------------------------ #
    # Divers                                                              #
    # ------------------------------------------------------------------ #

    def nb_etats(self) -> int:
        return len(self._probs)

    def vers_dict_deterministe(self) -> Dict[int, int]:
        """
        Vue compacte état -> action gloutonne, pratique pour l'affichage ou la
        comparaison de deux politiques déterministes.
        """
        return {s: self.meilleure_action(s) for s in range(len(self._probs))}

    def __eq__(self, autre: object) -> bool:
        if not isinstance(autre, Politique):
            return NotImplemented
        return self._probs == autre._probs

    def __repr__(self) -> str:
        return f"Politique(nb_etats={len(self._probs)})"
