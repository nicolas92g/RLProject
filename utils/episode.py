# Helper partagé : joue un épisode sur un ModelFreeEnv, récompense = delta score().
from typing import Callable, List, Tuple

from environments.base import ModelFreeEnv


def run_episode(
    env: ModelFreeEnv, choisir_action: Callable[[int], int]
) -> List[Tuple[int, int, float, int]]:
    # rejoue jusqu'à l'état terminal, choisir_action(etat) -> action (fournie par l'algo appelant)
    env.reset()
    transitions = []
    while not env.is_game_over():
        s = env.current_state()
        a = choisir_action(s)
        score_avant = env.score()
        env.step(a)
        r = env.score() - score_avant
        s_suivant = env.current_state()
        transitions.append((s, a, r, s_suivant))
    return transitions
