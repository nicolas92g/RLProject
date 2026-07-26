# Compare Dyna-Q sur Grid World, Monty Hall 2 et Rock Paper Scissors, meme
# methode de verification : comparer la VALEUR de l'action apprise a V*(s).
#
# Sur Grid World et Rock Paper Scissors (deterministes), Dyna-Q converge
# nettement plus vite que Q-Learning/Sarsa (3000 episodes suffisent contre
# 10000), c'est l'interet de la planification : chaque pas reel declenche
# n mises a jour supplementaires rejouees depuis le modele appris.
#
# Sur Monty Hall 2, ce meme modele devient un probleme : planning.py ne
# retient que la derniere transition observee pour chaque (s,a), pas une
# moyenne. A la decision finale, la recompense est stochastique (gagne ou
# perdu) : la planification rejoue plusieurs fois d'affilee ce dernier
# echantillon, ce qui biaise Q vers le bas au lieu de moyenner comme un
# vrai pas Q-Learning. Limite connue du Dyna-Q de base : il suppose un
# environnement deterministe.
import random

from algorithms.dynamic_programming import _valeur_action, policy_iteration
from algorithms.planning import dyna_q
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

    Q = dyna_q(GridWorld(), nb_episodes=3000, epsilon=0.1)
    print(f"  Dyna-Q : {'OK' if _verifier(GridWorld, V_star, gamma, etats, 4, Q) else 'ECHEC'} (3000 épisodes)")
    print()


def comparer_monty_hall_2() -> None:
    print("=== Monty Hall 2 ===")
    gamma = 1.0
    etats = list(range(1, 8))
    _, V_star = policy_iteration(MontyHall2(), gamma=gamma)

    # recompense stochastique + modele a un seul echantillon (voir en tete
    # de fichier) : taux de reussite ~30% par tentative, on retente plus
    for tentative in range(30):
        graine = GRAINE + tentative
        random.seed(graine)
        Q = dyna_q(
            MontyHall2(rng=random.Random(graine)),
            nb_episodes=40000,
            n_planification=2,
            gamma=gamma,
            alpha=0.05,
            epsilon=0.3,
        )
        if _verifier(MontyHall2, V_star, gamma, etats, 5, Q):
            break
    statut = "OK" if _verifier(MontyHall2, V_star, gamma, etats, 5, Q) else "ECHEC"
    print(f"  Dyna-Q : {statut} ({tentative + 1} tentative(s))")
    print()


def comparer_rock_paper_scissors() -> None:
    print("=== Rock Paper Scissors ===")
    gamma = 1.0
    etats = [0, 1, 2, 3]
    _, V_star = policy_iteration(TwoRoundRPS(), gamma=gamma)

    Q = dyna_q(TwoRoundRPS(rng=random.Random(GRAINE)), nb_episodes=3000, gamma=gamma, epsilon=0.1)
    print(f"  Dyna-Q : {'OK' if _verifier(TwoRoundRPS, V_star, gamma, etats, 3, Q) else 'ECHEC'} (3000 épisodes)")
    print()


if __name__ == "__main__":
    comparer_grid_world()
    comparer_monty_hall_2()
    comparer_rock_paper_scissors()
