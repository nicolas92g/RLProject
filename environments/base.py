# BASE definit les contrats avec 2 classes MDPenv et ModelFreeEnv, pour les environnements avec et sans modèle.

from typing import List


class MDPEnv : 
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
    
class ModelFreeEnv : 
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
    def pretty_print(self) -> None:
        raise NotImplementedError

    # pour des raisons de performance uniquement (pré-allocation de tableaux V/Q)
    def maximum_states_count(self) -> int:
        raise NotImplementedError
    def maximum_actions_count(self) -> int:
        raise NotImplementedError
    