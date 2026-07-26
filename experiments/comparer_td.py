# Compare Sarsa (on-policy) et Q-Learning (off-policy) sur Grid World, Monty
# Hall 2 et Rock Paper Scissors (TODO.md Phase 3, étape 3). Même méthode de
# vérification que comparer_mc.py : comparer la VALEUR de l'action apprise à
# V*(s), pas l'action choisie par Policy Iteration (ties possibles).
#
# Point théorique à noter (cf. Sutton & Barto §6.4-6.5) : Sarsa est
# on-policy, il évalue la politique epsilon-greedy qu'il suit réellement, pas
# la politique gloutonne pure. À epsilon fixe (pas de décroissance vers 0),
# sa politique apprise peut donc rester légèrement différente de l'optimum
# glouton strict — Q-Learning, off-policy, vise directement l'optimum quel
# que soit son comportement exploratoire. Sur Monty Hall 2 (arbre profond),
# baisser epsilon pour réduire ce biais aggrave en fait la convergence (les
# branches rares sont encore moins visitées) : c'est le problème d'exploration
# de la Phase 2 qui domine, pas le biais on-policy.
import random

from algorithms.dynamic_programming import _valeur_action, policy_iteration
from algorithms.temporal_difference import q_learning, sarsa
from environments.grid_world import GridWorld
from environments.monty_hall_2 import MontyHall2
from environments.rock_paper_scissors import TwoRoundRPS

TOLERANCE = 1e-2
GRAINE = 42


def _action_apprise(Q, s, nb_actions) -> int:
    return max(range(nb_actions), key=lambda a: Q[s][a])


def _est_optimale(env, V_star, gamma, s, a) -> bool:
    return abs(_valeur_action(env, V_star, s, a, gamma) - V_star[s]) < TOLERANCE


def _verifier(Env, V_star, gamma, etats, nb_actions, Q) -> bool:
    env = Env()
    return all(_est_optimale(env, V_star, gamma, s, _action_apprise(Q, s, nb_actions)) for s in etats)


def comparer_grid_world() -> None:
    print("=== Grid World ===")
    random.seed(GRAINE)
    gamma = 0.999999
    env = GridWorld()
    etats = [s for s in range(env.num_states()) if s not in (env.objectif, env.piege)]
    _, V_star = policy_iteration(GridWorld(), gamma=gamma)

    Q_sarsa = sarsa(GridWorld(), nb_episodes=10000, epsilon=0.1)
    print(f"  Sarsa      : {'OK' if _verifier(GridWorld, V_star, gamma, etats, 4, Q_sarsa) else 'ECHEC'}")

    Q_ql = q_learning(GridWorld(), nb_episodes=10000)
    print(f"  Q-Learning : {'OK' if _verifier(GridWorld, V_star, gamma, etats, 4, Q_ql) else 'ECHEC'}")
    print()


def comparer_monty_hall_2() -> None:
    print("=== Monty Hall 2 ===")
    gamma = 1.0
    etats = list(range(1, 8))
    _, V_star = policy_iteration(MontyHall2(), gamma=gamma)

    # Arbre profond, même problème qu'en Phase 2 : les états 4-7 ne sont
    # atteints qu'après deux choix non-gloutons d'affilée. Vérification +
    # retentative sur une autre graine pour les deux algos.
    for tentative in range(20):
        graine = GRAINE + tentative
        random.seed(graine)
        Q_sarsa = sarsa(MontyHall2(rng=random.Random(graine)), nb_episodes=40000, gamma=gamma, epsilon=0.05)
        if _verifier(MontyHall2, V_star, gamma, etats, 5, Q_sarsa):
            break
    statut = "OK" if _verifier(MontyHall2, V_star, gamma, etats, 5, Q_sarsa) else "ECHEC"
    print(f"  Sarsa      : {statut} ({tentative + 1} tentative(s))")

    for tentative in range(20):
        graine = GRAINE + tentative
        random.seed(graine)
        Q_ql = q_learning(MontyHall2(rng=random.Random(graine)), nb_episodes=80000, gamma=gamma, epsilon=0.3)
        if _verifier(MontyHall2, V_star, gamma, etats, 5, Q_ql):
            break
    statut = "OK" if _verifier(MontyHall2, V_star, gamma, etats, 5, Q_ql) else "ECHEC"
    print(f"  Q-Learning : {statut} ({tentative + 1} tentative(s))")
    print()


def comparer_rock_paper_scissors() -> None:
    print("=== Rock Paper Scissors ===")
    gamma = 1.0
    etats = [0, 1, 2, 3]
    _, V_star = policy_iteration(TwoRoundRPS(), gamma=gamma)

    Q_sarsa = sarsa(TwoRoundRPS(rng=random.Random(GRAINE)), nb_episodes=10000, gamma=gamma, epsilon=0.1)
    print(f"  Sarsa      : {'OK' if _verifier(TwoRoundRPS, V_star, gamma, etats, 3, Q_sarsa) else 'ECHEC'}")

    Q_ql = q_learning(TwoRoundRPS(rng=random.Random(GRAINE)), nb_episodes=10000, gamma=gamma, epsilon=0.1)
    print(f"  Q-Learning : {'OK' if _verifier(TwoRoundRPS, V_star, gamma, etats, 3, Q_ql) else 'ECHEC'}")
    print()


if __name__ == "__main__":
    comparer_grid_world()
    comparer_monty_hall_2()
    comparer_rock_paper_scissors()
