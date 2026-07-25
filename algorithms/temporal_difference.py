# Sarsa (on-policy) et Q-Learning (off-policy), cf. Sutton & Barto §6.4/6.5.
from typing import List

from algorithms.monte_carlo import action_epsilon_gloutonne, action_gloutonne
from environments.base import ModelFreeEnv


def sarsa(
    env: ModelFreeEnv, nb_episodes: int, gamma: float = 0.999999, alpha: float = 0.1, epsilon: float = 0.05
) -> List[List[float]]:
    nb_etats = env.maximum_states_count()
    nb_actions = env.maximum_actions_count()
    Q = [[0.0] * nb_actions for _ in range(nb_etats)]

    for _ in range(nb_episodes):
        env.reset()
        s = env.current_state()
        a = action_epsilon_gloutonne(Q, s, env.available_actions(), epsilon)

        while not env.is_game_over():
            score_avant = env.score()
            env.step(a)
            r = env.score() - score_avant
            s_p = env.current_state()

            if env.is_game_over():
                # terminal : pas de Q(s', a') à bootstrap, cf. convention Q[terminal,:]=0
                Q[s][a] += alpha * (r - Q[s][a])
                Q[s_p] = [0.0] * nb_actions
                break

            a_p = action_epsilon_gloutonne(Q, s_p, env.available_actions(), epsilon)
            Q[s][a] += alpha * (r + gamma * Q[s_p][a_p] - Q[s][a])
            s, a = s_p, a_p

    return Q


def q_learning(
    env: ModelFreeEnv, nb_episodes: int, gamma: float = 0.999999, alpha: float = 0.1, epsilon: float = 0.05
) -> List[List[float]]:
    nb_etats = env.maximum_states_count()
    nb_actions = env.maximum_actions_count()
    Q = [[0.0] * nb_actions for _ in range(nb_etats)]

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
                # off-policy : bootstrap sur le max, pas sur l'action réellement choisie ensuite
                meilleure_a = action_gloutonne(Q, s_p, env.available_actions())
                Q[s][a] += alpha * (r + gamma * Q[s_p][meilleure_a] - Q[s][a])

    return Q
