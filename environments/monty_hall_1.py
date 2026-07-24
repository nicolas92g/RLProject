# Monty Hall (3 portes, 2 décisions) : choisir une porte, puis garder/changer.
import random
from typing import List

from environments.base import MDPEnv, ModelFreeEnv

# États : 0 = départ, 1/2/3 = porte c=0/1/2 choisie (en attente garder/changer), 4 = terminal
# Actions : 0/1/2 au départ (choisir une porte), puis 0=garder / 1=changer (2 inutilisé)
# Le numéro de la porte révélée par le présentateur n'a pas besoin d'être dans l'état :
# à chosen_door fixé, P(gagner) ne dépend pas de laquelle des deux autres portes a été
# révélée (symétrie), donc (0/1/2/3/4) reste bien markovien pour la décision garder/changer.


class MontyHall1(MDPEnv):
    def num_states(self) -> int:
        return 5

    def num_actions(self) -> int:
        return 3

    def num_rewards(self) -> int:
        return 2

    def reward(self, i: int) -> float:
        return [0.0, 1.0][i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s == 0:
            # choisir une porte : pas de récompense, transition déterministe vers l'état 1+a
            if a in (0, 1, 2) and s_p == 1 + a and r_index == 0:
                return 1.0
            return 0.0
        elif s in (1, 2, 3):
            if s_p != 4:
                return 0.0
            if a == 0:  # garder : gagne si la porte choisie était la bonne (proba 1/3)
                if r_index == 1:
                    return 1.0 / 3.0
                return 2.0 / 3.0
            elif a == 1:  # changer : gagne dans les 2/3 des cas (résultat classique)
                if r_index == 1:
                    return 2.0 / 3.0
                return 1.0 / 3.0
            return 0.0
        else:  # s == 4, terminal
            return 0.0


class MontyHall1Env(ModelFreeEnv):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.winning_door = random.randint(0, 2)
        self.chosen_door = None
        self.revealed_door = None
        self.finished = False

    def is_game_over(self) -> bool:
        return self.finished

    def step(self, action: int) -> None:
        if self.is_game_over():
            raise ValueError("la partie est déjà terminée")
        if action not in self.available_actions():
            raise ValueError(f"action invalide : {action}")
        if self.chosen_door is None:
            self.chosen_door = action
            # le présentateur révèle une porte non choisie et non gagnante
            candidates = [d for d in range(3) if d != self.chosen_door and d != self.winning_door]
            self.revealed_door = random.choice(candidates)
        else:
            if action == 1:  # changer : la seule porte ni choisie ni révélée
                self.chosen_door = 3 - self.chosen_door - self.revealed_door
            self.finished = True

    def current_state(self) -> int:
        if self.chosen_door is None:
            return 0
        if not self.finished:
            return 1 + self.chosen_door
        return 4

    def available_actions(self) -> List[int]:
        if self.chosen_door is None:
            return [0, 1, 2]
        if not self.finished:
            return [0, 1]
        return []

    def score(self) -> float:
        if self.finished and self.chosen_door == self.winning_door:
            return 1.0
        return 0.0

    def pretty_print(self) -> None:
        cases = []
        for d in range(3):
            marque = "?"
            if d == self.revealed_door:
                marque = "X"
            elif self.finished and d == self.winning_door:
                marque = "G"
            elif d == self.chosen_door:
                marque = "C"
            cases.append(f"[{marque}]")
        print(" ".join(cases))

    def maximum_states_count(self) -> int:
        return 5

    def maximum_actions_count(self) -> int:
        return 3
