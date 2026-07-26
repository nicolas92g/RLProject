"""
Algorithmes Monte Carlo (necessitent ModelFreeEnv, taches episodiques).

Pas de reward() sur ModelFreeEnv : on prend la difference de score entre
deux appels a step().
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

from environments.base import ModelFreeEnv
from utils.policy import Politique

# un pas d'episode : etat, action, reward, proba de l'action (pour l'importance
# sampling), actions dispo a ce moment
EpisodeStep = Tuple[int, int, float, float, List[int]]


# Monte Carlo ES et on-policy first-visit MC control
#
# ModelFreeEnv n'a pas de vrai demarrage aleatoire, juste reset(). On approxime
# l'exploring start par une phase de pas aleatoires avant l'episode (cf FINDINGS.md)


def _demarrage_explorant(env: ModelFreeEnv, nb_pas_max: int) -> None:
    env.reset()
    for _ in range(random.randint(0, nb_pas_max)):
        if env.is_game_over():
            env.reset()
        env.step(random.choice(env.available_actions()))
    if env.is_game_over():
        env.reset()


def action_gloutonne(Q: List[List[float]], s: int, actions_disponibles: List[int]) -> int:
    # argmax restreint aux actions dispo (piege : ne pas prendre l'argmax sur
    # toutes les actions). Ex aequo tranches au hasard, sinon un Q a 0 partout
    # biaise vers l'action 0
    meilleure_valeur = max(Q[s][a] for a in actions_disponibles)
    gagnantes = [a for a in actions_disponibles if Q[s][a] == meilleure_valeur]
    return random.choice(gagnantes)


def action_epsilon_gloutonne(
    Q: List[List[float]], s: int, actions_disponibles: List[int], epsilon: float
) -> int:
    # exploration epsilon, sinon action gloutonne
    if random.random() < epsilon:
        return random.choice(actions_disponibles)
    return action_gloutonne(Q, s, actions_disponibles)


def _avertir_si_abandons(nb_abandonnes: int, nb_episodes: int, plafond: int) -> None:
    # episode sans etat terminal = retour G indefini, on l'ignore mais on le signale
    if nb_abandonnes > 0:
        part = 100.0 * nb_abandonnes / nb_episodes
        print(
            f"ATTENTION : {nb_abandonnes}/{nb_episodes} episodes ({part:.0f}%) abandonnes, "
            f"terminal jamais atteint en {plafond} pas, augmenter nb_pas_episode_max"
        )


def _premiere_visite(transitions: List[tuple]) -> List[bool]:
    # marque la premiere occurrence de chaque (s,a) dans l'episode
    vus = set()
    marques = []
    for s, a, _ in transitions:
        marques.append((s, a) not in vus)
        vus.add((s, a))
    return marques


def monte_carlo_es(
    env: ModelFreeEnv,
    nb_episodes: int,
    gamma: float = 0.999999,
    nb_pas_echauffement_max: int = 10,
    nb_pas_episode_max: int = 10_000,
) -> List[List[float]]:
    nb_etats = env.maximum_states_count()
    nb_actions = env.maximum_actions_count()
    Q = [[0.0] * nb_actions for _ in range(nb_etats)]
    compte = [[0] * nb_actions for _ in range(nb_etats)]
    nb_abandonnes = 0

    for _ in range(nb_episodes):
        _demarrage_explorant(env, nb_pas_echauffement_max)
        a = random.choice(env.available_actions())  # exploring start

        # sans exploration apres le premier pas, la politique gloutonne peut
        # boucler indefiniment entre 2 etats : filet de securite
        transitions = []
        for _ in range(nb_pas_episode_max):
            s = env.current_state()
            score_avant = env.score()
            env.step(a)
            r = env.score() - score_avant
            transitions.append((s, a, r))
            if env.is_game_over():
                break
            s_suivant = env.current_state()
            a = action_gloutonne(Q, s_suivant, env.available_actions())
        else:
            nb_abandonnes += 1
            continue

        marques = _premiere_visite(transitions)
        G = 0.0
        for t in range(len(transitions) - 1, -1, -1):
            s, a, r = transitions[t]
            G = gamma * G + r
            if marques[t]:
                compte[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / compte[s][a]

        Q[env.current_state()] = [0.0] * nb_actions  # etat terminal : Q = 0

    _avertir_si_abandons(nb_abandonnes, nb_episodes, nb_pas_episode_max)
    return Q


def mc_on_policy_first_visit(
    env: ModelFreeEnv,
    nb_episodes: int,
    gamma: float = 0.999999,
    epsilon: float = 0.1,
    nb_pas_episode_max: int = 10_000,
) -> List[List[float]]:
    nb_etats = env.maximum_states_count()
    nb_actions = env.maximum_actions_count()
    Q = [[0.0] * nb_actions for _ in range(nb_etats)]
    compte = [[0] * nb_actions for _ in range(nb_etats)]
    nb_abandonnes = 0

    for _ in range(nb_episodes):
        env.reset()
        transitions = []
        for _ in range(nb_pas_episode_max):
            if env.is_game_over():
                break
            s = env.current_state()
            a = action_epsilon_gloutonne(Q, s, env.available_actions(), epsilon)
            score_avant = env.score()
            env.step(a)
            r = env.score() - score_avant
            transitions.append((s, a, r))
        else:
            nb_abandonnes += 1
            continue

        marques = _premiere_visite(transitions)
        G = 0.0
        for t in range(len(transitions) - 1, -1, -1):
            s, a, r = transitions[t]
            G = gamma * G + r
            if marques[t]:
                compte[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / compte[s][a]

        Q[env.current_state()] = [0.0] * nb_actions

    _avertir_si_abandons(nb_abandonnes, nb_episodes, nb_pas_episode_max)
    return Q


# Off-policy MC control, importance sampling pondere
#
# On veut apprendre la politique gloutonne optimale (la cible), mais si on ne
# joue que des coups gloutons on n'explore jamais les actions mal estimees. On
# genere les episodes avec une politique de comportement b (epsilon-greedy),
# et on corrige l'ecart entre b et la cible via un poids d'importance W.


def _behavior_policy(
    q_s: Sequence[float], available_actions: List[int], epsilon: float
) -> Dict[int, float]:
    # distribution epsilon-greedy de b, restreinte aux actions dispo
    n = len(available_actions)
    q_max = max(q_s[a] for a in available_actions)
    best_actions = [a for a in available_actions if q_s[a] == q_max]
    probs = {}
    for a in available_actions:
        p = epsilon / n
        if a in best_actions:
            p += (1.0 - epsilon) / len(best_actions)
        probs[a] = p
    return probs


def _greedy_action(q_s: Sequence[float], available_actions: List[int]) -> int:
    # action cible : argmax de Q parmi les actions dispo
    return max(available_actions, key=lambda a: q_s[a])


def _sample_action(probs: Dict[int, float], rng: random.Random) -> int:
    # tire une action selon la distribution probs
    draw = rng.random()
    cumulative = 0.0
    for a, p in probs.items():
        cumulative += p
        if draw < cumulative:
            return a
    return next(reversed(probs.keys()))  # filet contre l'imprecision flottante


def _generate_episode(
    env: ModelFreeEnv,
    q: List[List[float]],
    epsilon: float,
    rng: random.Random,
) -> List[EpisodeStep]:
    # joue un episode complet en suivant b
    env.reset()
    episode: List[EpisodeStep] = []
    while not env.is_game_over():
        s = env.current_state()
        available = env.available_actions()
        probs = _behavior_policy(q[s], available, epsilon)
        a = _sample_action(probs, rng)
        prev_score = env.score()
        env.step(a)
        r = env.score() - prev_score
        episode.append((s, a, r, probs[a], available))
    return episode


def off_policy_mc_control(
    env: ModelFreeEnv,
    n_episodes: int,
    gamma: float = 0.999,
    epsilon: float = 0.1,
    rng: Optional[random.Random] = None,
) -> Tuple[Politique, List[List[float]]]:
    # off-policy MC control, importance sampling pondere
    # b (comportement) : epsilon-greedy sur Q, regeneree a chaque pas
    # pi (cible) : gloutonne sur Q, c'est elle qu'on renvoie
    rng = rng if rng is not None else random.Random()
    n_states = env.maximum_states_count()
    n_actions = env.maximum_actions_count()

    q = [[0.0] * n_actions for _ in range(n_states)]
    c = [[0.0] * n_actions for _ in range(n_states)]  # C(s,a) : somme des poids d'importance

    for _ in range(n_episodes):
        episode = _generate_episode(env, q, epsilon, rng)

        g = 0.0  # retour actualise
        w = 1.0  # poids d'importance
        for s, a, r, b_prob, available in reversed(episode):
            g = gamma * g + r
            c[s][a] += w
            q[s][a] += (w / c[s][a]) * (g - q[s][a])
            # pi gloutonne : des que b devie de pi, le reste de la trajectoire
            # (plus ancien) n'informe plus sur pi, on passe a l'episode suivant
            if a != _greedy_action(q[s], available):
                break
            w /= b_prob  # pi(a|s) = 1 sur l'action gloutonne, W ne divise que par b

    return Politique.gloutonne(q), q


# validation rapide sur Grid World : la politique doit mener a G en evitant X
# python -m algorithms.monte_carlo
if __name__ == "__main__":
    from environments.grid_world import GridWorld, NOMS_ACTIONS

    env = GridWorld()
    policy, q = off_policy_mc_control(
        env, n_episodes=20000, gamma=0.999, epsilon=0.2, rng=random.Random(0)
    )

    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    print("Politique gloutonne apprise (off-policy MC control) :\n")
    for row in range(env.nb_lignes):
        cells = []
        for col in range(env.nb_colonnes):
            s = row * env.nb_colonnes + col
            if s == env.objectif:
                cells.append("G")
            elif s == env.piege:
                cells.append("X")
            else:
                cells.append(arrows[policy.meilleure_action(s)])
        print("  " + " ".join(cells))

    print("\nQ(depart) par action :")
    for a in range(env.num_actions()):
        print(f"  {NOMS_ACTIONS[a]:<7} : {q[env.depart][a]:+.4f}")
