from typing import List

from environments.base import MDPEnv, ModelFreeEnv

class LineWorld(MDPEnv):
    def num_states (self) -> int:
        return 5
    def num_actions(self) -> int:
        return 2
    def num_rewards(self) -> int:
        return 3
    def reward(self, i: int) -> float:
        # indexation du notebook du prof : 0 = neutre, 1 = perdu, 2 = gagné
        rewards = [0.0, -1.0, 1.0]
        return rewards[i]
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s == 0:
            return 0.0
        elif s == 1:
            if a == 0 and s_p == 0 and r_index == 1:
                return 1.0
            elif a == 1 and s_p == 2 and r_index == 0:
                return 1.0
            else:
                return 0.0
        elif s == 2:
            if a == 0 and s_p == 1 and r_index == 0:
                return 1.0
            elif a == 1 and s_p == 3 and r_index == 0:
                return 1.0
            else:
                return 0.0
        elif s == 3:
            if a == 0 and s_p == 2 and r_index == 0:
                return 1.0
            elif a == 1 and s_p == 4 and r_index == 2:
                return 1.0
            else:
                return 0.0
        elif s == 4:
            return 0.0
        else : return 0.0


class LineWorldEnv(ModelFreeEnv):
    # conventions du notebook du prof : 5 cases, départ au milieu, terminal aux bouts
    NUM_CELLS = 5

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.agent_pos = self.NUM_CELLS // 2

    def is_game_over(self) -> bool:
        return self.agent_pos == 0 or self.agent_pos == self.NUM_CELLS - 1

    def step(self, action: int) -> None:
        if self.is_game_over():
            raise ValueError("la partie est déjà terminée")
        if action not in self.available_actions():
            raise ValueError(f"action invalide : {action}")
        if action == 0:
            self.agent_pos -= 1
        else:
            self.agent_pos += 1

    def current_state(self) -> int:
        return self.agent_pos

    def available_actions(self) -> List[int]:
        return [0, 1]

    def score(self) -> float:
        if self.agent_pos == 0:
            return -1.0
        if self.agent_pos == self.NUM_CELLS - 1:
            return 1.0
        return 0.0

    def pretty_print(self) -> None:
        cases = ["_"] * self.NUM_CELLS
        cases[self.agent_pos] = "X"
        print("".join(cases))

    def maximum_states_count(self) -> int:
        return self.NUM_CELLS

    def maximum_actions_count(self) -> int:
        return 2