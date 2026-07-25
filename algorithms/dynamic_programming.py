from typing import List, Tuple

from environments.base import MDPEnv
from utils.policy import Politique


def _valeur_action(env: MDPEnv, V: List[float], s: int, a: int, gamma: float) -> float:
    # espérance de r + gamma * V(s') sur (s', r), cf. Sutton & Barto §4.1
    total = 0.0
    for s_p in range(env.num_states()):
        for r_index in range(env.num_rewards()):
            proba = env.p(s, a, s_p, r_index)
            if proba > 0.0:
                total += proba * (env.reward(r_index) + gamma * V[s_p])
    return total


def _evaluation_politique(env: MDPEnv, pi: Politique, V: List[float], gamma: float, theta: float) -> None:
    # met à jour V en place jusqu'à convergence, cf. Sutton & Barto eq. 4.5
    nb_etats = env.num_states()
    nb_actions = env.num_actions()
    while True:
        delta = 0.0
        for s in range(nb_etats):
            ancienne_valeur = V[s]
            nouvelle_valeur = 0.0
            for a in range(nb_actions):
                proba_action = pi.proba(s, a)
                if proba_action > 0.0:
                    nouvelle_valeur += proba_action * _valeur_action(env, V, s, a, gamma)
            V[s] = nouvelle_valeur
            delta = max(delta, abs(ancienne_valeur - V[s]))
        if delta < theta:
            break


def _amelioration_politique(env: MDPEnv, V: List[float], gamma: float) -> Tuple[Politique, List[int]]:
    # extrait l'action gloutonne par état, cf. Sutton & Barto eq. 4.9
    nb_etats = env.num_states()
    nb_actions = env.num_actions()
    meilleures_actions = []
    for s in range(nb_etats):
        meilleure_action = 0
        meilleur_score = float("-inf")
        for a in range(nb_actions):
            score = _valeur_action(env, V, s, a, gamma)
            if score > meilleur_score:
                meilleur_score = score
                meilleure_action = a
        meilleures_actions.append(meilleure_action)
    return Politique.deterministe(meilleures_actions, nb_actions), meilleures_actions


def policy_iteration(
    env: MDPEnv, gamma: float = 0.999999, theta: float = 0.000001
) -> Tuple[Politique, List[float]]:
    # alternance évaluation / amélioration jusqu'à politique stable, cf. Sutton & Barto §4.3
    nb_etats = env.num_states()
    V = [0.0] * nb_etats
    pi = Politique.uniforme(nb_etats, env.num_actions())
    actions_precedentes = [pi.meilleure_action(s) for s in range(nb_etats)]

    while True:
        _evaluation_politique(env, pi, V, gamma, theta)
        pi, actions_courantes = _amelioration_politique(env, V, gamma)
        if actions_courantes == actions_precedentes:
            break
        actions_precedentes = actions_courantes

    return pi, V


def value_iteration(
    env: MDPEnv, gamma: float = 0.999999, theta: float = 0.000001
) -> Tuple[Politique, List[float]]:
    # évaluation et amélioration fusionnées via un max par sweep, cf. Sutton & Barto eq. 4.10
    nb_etats = env.num_states()
    nb_actions = env.num_actions()
    V = [0.0] * nb_etats

    while True:
        delta = 0.0
        for s in range(nb_etats):
            ancienne_valeur = V[s]
            V[s] = max(_valeur_action(env, V, s, a, gamma) for a in range(nb_actions))
            delta = max(delta, abs(ancienne_valeur - V[s]))
        if delta < theta:
            break

    pi, _ = _amelioration_politique(env, V, gamma)
    return pi, V
