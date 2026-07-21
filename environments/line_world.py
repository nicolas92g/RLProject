from environments.base import MDPEnv

class LineWorld(MDPEnv):
    def num_states (self) -> int:
        return 5
    def num_actions(self) -> int:
        return 2
    def num_rewards(self) -> int:
        return 3
    def reward(self, i: int) -> float:
        rewards = [-1.0, 0.0, 1.0]
        return rewards[i]
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s == 0:
            if a == 0 and s_p == 0 and r_index == 0:
                return 1.0
            elif a == 1 and s_p == 1 and r_index == 1:
                return 1.0
            else:
                return 0.0
        elif s == 1:
            if a == 0 and s_p == 0 and r_index == 0:
                return 1.0
            elif a == 1 and s_p == 2 and r_index == 1:
                return 1.0
            else:
                return 0.0
        elif s == 2:
            if a == 0 and s_p == 1 and r_index == 1:
                return 1.0
            elif a == 1 and s_p == 3 and r_index == 1:
                return 1.0
            else:
                return 0.0
        elif s == 3:
            if a == 0 and s_p == 2 and r_index == 1:
                return 1.0
            elif a == 1 and s_p == 4 and r_index == 2:
                return 1.0
            else:
                return 0.0
        elif s == 4:
            if a in [0,1] and s_p == 4 and r_index in [2]:
                return 1.0
            else:
                return 0.0