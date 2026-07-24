# Monte Carlo ES et On-policy first-visit MC Control, cf. Sutton & Barto §5.3/§5.4.
import random
from typing import List

from environments.base import ModelFreeEnv

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
    gamma: float = 1.0,
    nb_pas_echauffement_max: int = 10,
    nb_pas_episode_max: int = 200,
) -> List[List[float]]:
    nb_etats = env.maximum_states_count()
    nb_actions = env.maximum_actions_count()
    Q = [[0.0] * nb_actions for _ in range(nb_etats)]
    compte = [[0] * nb_actions for _ in range(nb_etats)]

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
            continue  # cycle détecté (pas de terminal atteint) : épisode ignoré

        marques = _premiere_visite(transitions)
        G = 0.0
        for t in range(len(transitions) - 1, -1, -1):
            s, a, r = transitions[t]
            G = gamma * G + r
            if marques[t]:
                compte[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / compte[s][a]

        Q[env.current_state()] = [0.0] * nb_actions  # état terminal : Q = 0

    return Q


def mc_on_policy_first_visit(
    env: ModelFreeEnv,
    nb_episodes: int,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    nb_pas_episode_max: int = 200,
) -> List[List[float]]:
    nb_etats = env.maximum_states_count()
    nb_actions = env.maximum_actions_count()
    Q = [[0.0] * nb_actions for _ in range(nb_etats)]
    compte = [[0] * nb_actions for _ in range(nb_etats)]

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
            continue  # filet de sécurité, cf. monte_carlo_es (peu probable en epsilon-soft)

        marques = _premiere_visite(transitions)
        G = 0.0
        for t in range(len(transitions) - 1, -1, -1):
            s, a, r = transitions[t]
            G = gamma * G + r
            if marques[t]:
                compte[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / compte[s][a]

        Q[env.current_state()] = [0.0] * nb_actions  # état terminal : Q = 0

    return Q
