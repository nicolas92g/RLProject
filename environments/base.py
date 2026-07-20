from typing import List


class MDPEnv:
    """
    Contrat pour un environnement exposant son modèle complet
    (transitions et récompenses connues à l'avance).

    Nécessaire pour les algorithmes de Dynamic Programming
    (Policy Iteration, Value Iteration).
    """

    def num_states(self) -> int:
        raise NotImplementedError

    def num_actions(self) -> int:
        raise NotImplementedError

    def num_rewards(self) -> int:
        raise NotImplementedError

    def reward(self, i: int) -> float:
        raise NotImplementedError

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        raise NotImplementedError


class ModelFreeEnv:
    """
    Contrat pour un environnement sans modèle : l'agent apprend
    uniquement par interaction (reset/step).

    Nécessaire pour Monte Carlo, TD Learning (Sarsa, Q-Learning)
    et Planning (Dyna-Q, Dyna-Q+).
    """

    def reset(self) -> None:
        raise NotImplementedError

    def step(self, action: int) -> None:
        raise NotImplementedError

    def is_game_over(self) -> bool:
        raise NotImplementedError

    def current_state(self) -> int:
        raise NotImplementedError

    def available_actions(self) -> List[int]:
        raise NotImplementedError

    def score(self) -> float:
        raise NotImplementedError

    # pour des raisons de performance uniquement (pré-allocation de tableaux V/Q)
    def maximum_states_count(self) -> int:
        raise NotImplementedError

    def maximum_actions_count(self) -> int:
        raise NotImplementedError

    # optionnel
    def pretty_print(self) -> None:
        raise NotImplementedError