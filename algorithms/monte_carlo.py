"""
Algorithmes Monte Carlo (nécessitent `ModelFreeEnv`, tâches épisodiques).

Convention projet (cf. CLAUDE.md) : pas de méthode `reward()` sur
`ModelFreeEnv`. La récompense d'un pas se reconstitue par différence de score :

    prev = env.score()
    env.step(a)
    r = env.score() - prev
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

from environments.base import ModelFreeEnv
from utils.policy import Politique

# Un pas d'épisode : état, action jouée, récompense reçue, probabilité sous
# laquelle l'action a été tirée (pour l'importance sampling) et actions qui
# étaient disponibles à ce moment-là.
EpisodeStep = Tuple[int, int, float, float, List[int]]


# ----------------------------------------------------------------------------
# Off-policy MC Control (importance sampling pondéré)
# ----------------------------------------------------------------------------
#
# Idée : on veut apprendre la politique gloutonne optimale π (la *cible*), mais
# si on ne jouait que des coups gloutons on n'explorerait jamais les actions
# encore mal estimées. On génère donc les épisodes avec une politique *de
# comportement* b, ε-greedy (donc "soft" : toute action a une proba > 0), et on
# corrige le décalage entre ce que b a joué et ce que π aurait joué au moyen du
# poids d'importance W. C'est le pendant "sans modèle" de Value Iteration :
# apprendre l'optimum tout en explorant avec autre chose que l'optimum.


def _behavior_policy(
    q_s: Sequence[float], available_actions: List[int], epsilon: float
) -> Dict[int, float]:
    """
    Distribution ε-greedy de la politique de comportement b, restreinte aux
    seules actions disponibles dans l'état (indispensable pour Monty Hall &
    consorts, où le jeu d'actions change d'un état à l'autre).
    """
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
    """Action cible : argmax de Q parmi les actions disponibles (π est gloutonne)."""
    return max(available_actions, key=lambda a: q_s[a])


def _sample_action(probs: Dict[int, float], rng: random.Random) -> int:
    """Tire une action selon la distribution `probs`."""
    draw = rng.random()
    cumulative = 0.0
    for a, p in probs.items():
        cumulative += p
        if draw < cumulative:
            return a
    return next(reversed(probs.keys()))  # filet contre l'imprécision flottante


def _generate_episode(
    env: ModelFreeEnv,
    q: List[List[float]],
    epsilon: float,
    rng: random.Random,
) -> List[EpisodeStep]:
    """Joue un épisode complet en suivant la politique de comportement b."""
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
    """
    Off-policy MC Control par importance sampling pondéré (Sutton & Barto, §5.7).

    Retourne la politique cible gloutonne apprise et la table Q correspondante.

    - `b` (comportement) : ε-greedy sur Q, régénérée à chaque pas d'épisode.
    - `π` (cible) : gloutonne sur Q, c'est elle qu'on renvoie.
    """
    rng = rng if rng is not None else random.Random()
    n_states = env.maximum_states_count()
    n_actions = env.maximum_actions_count()

    q = [[0.0] * n_actions for _ in range(n_states)]
    # C(s,a) : somme cumulée des poids d'importance, dénominateur de la
    # moyenne pondérée (weighted importance sampling).
    c = [[0.0] * n_actions for _ in range(n_states)]

    for _ in range(n_episodes):
        episode = _generate_episode(env, q, epsilon, rng)

        g = 0.0  # retour actualisé, accumulé en remontant l'épisode
        w = 1.0  # poids d'importance courant
        for s, a, r, b_prob, available in reversed(episode):
            g = gamma * g + r
            c[s][a] += w
            q[s][a] += (w / c[s][a]) * (g - q[s][a])
            # π est gloutonne : dès que b a dévié de π, le reste de la
            # trajectoire (plus ancien) n'informe plus sur π → épisode suivant.
            if a != _greedy_action(q[s], available):
                break
            # π(a|s) = 1 sur l'action gloutonne, donc W ne divise que par b(a|s).
            w /= b_prob

    return Politique.gloutonne(q), q


# Validation rapide sur Grid World : la politique apprise doit mener à G (+1)
# en évitant X (-1). Lancer : python -m algorithms.monte_carlo
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