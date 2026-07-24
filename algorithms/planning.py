# Dyna-Q, cf. Sutton & Barto §8.2.
import random
from typing import Dict, List, Tuple

from algorithms.monte_carlo import action_epsilon_gloutonne
from environments.base import ModelFreeEnv

# Le max_a Q(s',a) ne peut pas interroger available_actions() pour un état simulé
# (l'env réel n'y est pas physiquement) : on le restreint aux actions déjà observées
# dans cet état. Indispensable dès que les récompenses peuvent être négatives (secret
# envs) : une action inexistante garde Q = 0.0, qui l'emporterait sur un vrai Q négatif
# et surestimerait la valeur de l'état. Voir FINDINGS.md.


def _max_q(Q, s: int, actions_vues: dict) -> float:
    # max restreint aux actions déjà prises dans s ; 0.0 si l'état n'a jamais été visité
    connues = actions_vues.get(s)
    if not connues:
        return 0.0
    return max(Q[s][a] for a in connues)


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

            modele[(s, a)] = (r, s_p)
            if a not in actions_vues.setdefault(s, []):
                actions_vues[s].append(a)

            if env.is_game_over():
                Q[s][a] += alpha * (r - Q[s][a])
                Q[s_p] = [0.0] * nb_actions
            else:
                # bootstrap sur le max restreint aux actions connues de s', cf. _max_q
                Q[s][a] += alpha * (r + gamma * _max_q(Q, s_p, actions_vues) - Q[s][a])

            # planification : n mises à jour supplémentaires simulées depuis le modèle appris
            etats_connus = list(actions_vues.keys())
            for _ in range(n_planification):
                s_sim = random.choice(etats_connus)
                a_sim = random.choice(actions_vues[s_sim])
                r_sim, s_p_sim = modele[(s_sim, a_sim)]
                Q[s_sim][a_sim] += alpha * (
                    r_sim + gamma * _max_q(Q, s_p_sim, actions_vues) - Q[s_sim][a_sim]
                )

    return Q
