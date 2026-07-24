# Dyna-Q, cf. Sutton & Barto §8.2.
import random
from typing import Dict, List, Tuple

from algorithms.monte_carlo import action_epsilon_gloutonne
from environments.base import ModelFreeEnv

# Limite connue : les mises à jour de planification ne peuvent pas interroger
# available_actions() pour un état simulé (l'env réel n'y est pas physiquement) : le
# max_a Q(s',a) y porte sur tout l'espace d'actions, pas restreint. Sans incidence
# mesurée sur LineWorld/Monty Hall (les actions valides y gardent des Q positifs qui
# dominent une action jamais explorée, restée à 0.0) — documenté dans FINDINGS.md.


def dyna_q(
    env: ModelFreeEnv,
    nb_episodes: int,
    n_planification: int = 10,
    gamma: float = 0.999999,
    alpha: float = 0.1,
    epsilon: float = 0.05,
) -> List[List[float]]:
    nb_etats = env.maximum_states_count()
    nb_actions = env.maximum_actions_count()
    Q = [[0.0] * nb_actions for _ in range(nb_etats)]
    modele: Dict[Tuple[int, int], Tuple[float, int]] = {}
    actions_vues: Dict[int, list] = {}  # état déjà observé -> actions déjà prises dedans

    for _ in range(nb_episodes):
        env.reset()
        while not env.is_game_over():
            s = env.current_state()
            a = action_epsilon_gloutonne(Q, s, env.available_actions(), epsilon)
            score_avant = env.score()
            env.step(a)
            r = env.score() - score_avant
            s_p = env.current_state()

            if env.is_game_over():
                Q[s][a] += alpha * (r - Q[s][a])
                Q[s_p] = [0.0] * nb_actions
            else:
                Q[s][a] += alpha * (r + gamma * max(Q[s_p]) - Q[s][a])

            modele[(s, a)] = (r, s_p)
            if a not in actions_vues.setdefault(s, []):
                actions_vues[s].append(a)

            # planification : n mises à jour supplémentaires simulées depuis le modèle appris
            etats_connus = list(actions_vues.keys())
            for _ in range(n_planification):
                s_sim = random.choice(etats_connus)
                a_sim = random.choice(actions_vues[s_sim])
                r_sim, s_p_sim = modele[(s_sim, a_sim)]
                Q[s_sim][a_sim] += alpha * (r_sim + gamma * max(Q[s_p_sim]) - Q[s_sim][a_sim])

    return Q
