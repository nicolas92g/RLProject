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
# Monte Carlo ES et On-policy first-visit MC Control, cf. Sutton & Barto §5.3/§5.4.
# ----------------------------------------------------------------------------
#
# Le contrat ModelFreeEnv n'offre pas de "démarrage dans un état arbitraire" (juste
# reset()) : on approxime l'exploring start par une phase d'actions aléatoires avant
# l'épisode suivi, hypothèse documentée (FINDINGS.md) faute d'un vrai from_random_state
# générique dans le contrat.


def _demarrage_explorant(env: ModelFreeEnv, nb_pas_max: int) -> None:
    env.reset()
    for _ in range(random.randint(0, nb_pas_max)):
        if env.is_game_over():
            env.reset()
        env.step(random.choice(env.available_actions()))
    if env.is_game_over():
        env.reset()


def action_gloutonne(Q: List[List[float]], s: int, actions_disponibles: List[int]) -> int:
    # argmax restreint aux actions disponibles dans cet état (piège connu : ne pas
    # argmax-er sur l'espace d'actions complet, cf. FINDINGS.md is_forbidden/Monty Hall).
    # Ex æquo tranchés au hasard (pas toujours la première action), cf. Politique.gloutonne :
    # sinon un Q initialisé à 0 partout biaise systématiquement vers l'action 0.
    meilleure_valeur = max(Q[s][a] for a in actions_disponibles)
    gagnantes = [a for a in actions_disponibles if Q[s][a] == meilleure_valeur]
    return random.choice(gagnantes)


def action_epsilon_gloutonne(
    Q: List[List[float]], s: int, actions_disponibles: List[int], epsilon: float
) -> int:
    # exploration epsilon, sinon action gloutonne restreinte (cf. action_gloutonne)
    if random.random() < epsilon:
        return random.choice(actions_disponibles)
    return action_gloutonne(Q, s, actions_disponibles)


def _avertir_si_abandons(nb_abandonnes: int, nb_episodes: int, plafond: int) -> None:
    # un épisode sans état terminal n'a pas de retour G défini : on ne peut que l'ignorer,
    # mais jamais en silence — un plafond trop bas vide Q sans que rien ne le signale
    if nb_abandonnes > 0:
        part = 100.0 * nb_abandonnes / nb_episodes
        print(
            f"ATTENTION : {nb_abandonnes}/{nb_episodes} épisodes ({part:.0f}%) abandonnés, "
            f"terminal jamais atteint en {plafond} pas — augmenter nb_pas_episode_max"
        )


def _premiere_visite(transitions: List[tuple]) -> List[bool]:
    # marque, pour chaque pas, s'il s'agit de la première occurrence de (s,a) dans l'épisode
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
        a = random.choice(env.available_actions())  # exploring start : action initiale forcée

        # sans exploration après le premier pas, la politique gloutonne courante peut
        # cycler indéfiniment entre deux états (jamais de terminal) : filet de sécurité
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
            nb_abandonnes += 1  # pas de terminal atteint : retour G indéfini, épisode ignoré
            continue

        marques = _premiere_visite(transitions)
        G = 0.0
        for t in range(len(transitions) - 1, -1, -1):
            s, a, r = transitions[t]
            G = gamma * G + r
            if marques[t]:
                compte[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / compte[s][a]

        Q[env.current_state()] = [0.0] * nb_actions  # état terminal : Q = 0

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
            # politique epsilon-soft dérivée directement de Q, cf. Sutton & Barto §5.4
            a = action_epsilon_gloutonne(Q, s, env.available_actions(), epsilon)
            score_avant = env.score()
            env.step(a)
            r = env.score() - score_avant
            transitions.append((s, a, r))
        else:
            nb_abandonnes += 1  # cf. monte_carlo_es
            continue

        marques = _premiere_visite(transitions)
        G = 0.0
        for t in range(len(transitions) - 1, -1, -1):
            s, a, r = transitions[t]
            G = gamma * G + r
            if marques[t]:
                compte[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / compte[s][a]

        Q[env.current_state()] = [0.0] * nb_actions  # état terminal : Q = 0

    _avertir_si_abandons(nb_abandonnes, nb_episodes, nb_pas_episode_max)
    return Q


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
