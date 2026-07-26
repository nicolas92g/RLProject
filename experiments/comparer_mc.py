# Compare Monte Carlo ES, on-policy first-visit et off-policy MC control sur
# Grid World, Monty Hall 2 et Rock Paper Scissors, verifie contre l'optimum
# exact (Policy Iteration). On compare la VALEUR de l'action apprise a
# V*(s), pas l'action choisie par Policy Iteration : plusieurs de ces
# environnements ont des actions ex-aequo, comparer les indices serait trop
# strict.
import random

from algorithms.dynamic_programming import _valeur_action, policy_iteration
from algorithms.monte_carlo import mc_on_policy_first_visit, monte_carlo_es, off_policy_mc_control
from environments.grid_world import GridWorld
from environments.monty_hall_2 import MontyHall2
from environments.rock_paper_scissors import TwoRoundRPS

TOLERANCE = 1e-3
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
    # GridWorld n'a pas de rng injectable : on fixe la graine globale pour
    # rendre le taux d'abandon de MC ES reproductible
    random.seed(GRAINE)
    gamma = 0.999999
    env = GridWorld()
    etats = [s for s in range(env.num_states()) if s not in (env.objectif, env.piege)]
    _, V_star = policy_iteration(GridWorld(), gamma=gamma)

    Q_es = monte_carlo_es(GridWorld(), nb_episodes=10000)
    print(f"  MC ES          : {'OK' if _verifier(GridWorld, V_star, gamma, etats, 4, Q_es) else 'ECHEC'}")

    Q_on = mc_on_policy_first_visit(GridWorld(), nb_episodes=10000, epsilon=0.1)
    print(f"  MC on-policy   : {'OK' if _verifier(GridWorld, V_star, gamma, etats, 4, Q_on) else 'ECHEC'}")

    _, Q_off = off_policy_mc_control(
        GridWorld(), n_episodes=10000, gamma=gamma, epsilon=0.1, rng=random.Random(GRAINE)
    )
    print(f"  MC off-policy  : {'OK' if _verifier(GridWorld, V_star, gamma, etats, 4, Q_off) else 'ECHEC'}")
    print()


def comparer_monty_hall_2() -> None:
    print("=== Monty Hall 2 ===")
    gamma = 1.0
    etats = list(range(1, 8))
    _, V_star = policy_iteration(MontyHall2(), gamma=gamma)

    # arbre profond : les etats 4-7 ne sont atteints qu'apres 2 choix
    # non-gloutons d'affilee, epsilon releve + retentative pour les 2 algos
    # epsilon-greedy. MC ES n'a pas ce probleme (exploring starts)

    Q_es = monte_carlo_es(
        MontyHall2(rng=random.Random(GRAINE)), nb_episodes=40000, gamma=gamma, nb_pas_echauffement_max=4
    )
    print(f"  MC ES          : {'OK' if _verifier(MontyHall2, V_star, gamma, etats, 5, Q_es) else 'ECHEC'}")

    for tentative in range(20):
        graine = GRAINE + tentative
        random.seed(graine)
        Q_on = mc_on_policy_first_visit(
            MontyHall2(rng=random.Random(graine)), nb_episodes=40000, gamma=gamma, epsilon=0.2
        )
        if _verifier(MontyHall2, V_star, gamma, etats, 5, Q_on):
            break
    statut = "OK" if _verifier(MontyHall2, V_star, gamma, etats, 5, Q_on) else "ECHEC"
    print(f"  MC on-policy   : {statut} ({tentative + 1} tentative(s))")

    for tentative in range(20):
        graine = GRAINE + tentative
        _, Q_off = off_policy_mc_control(
            MontyHall2(rng=random.Random(graine)),
            n_episodes=40000,
            gamma=gamma,
            epsilon=0.2,
            rng=random.Random(graine + 1),
        )
        if _verifier(MontyHall2, V_star, gamma, etats, 5, Q_off):
            break
    statut = "OK" if _verifier(MontyHall2, V_star, gamma, etats, 5, Q_off) else "ECHEC"
    print(f"  MC off-policy  : {statut} ({tentative + 1} tentative(s))")
    print()


def comparer_rock_paper_scissors() -> None:
    print("=== Rock Paper Scissors ===")
    gamma = 1.0
    etats = [0, 1, 2, 3]
    _, V_star = policy_iteration(TwoRoundRPS(), gamma=gamma)

    Q_es = monte_carlo_es(
        TwoRoundRPS(rng=random.Random(GRAINE)), nb_episodes=10000, gamma=gamma, nb_pas_echauffement_max=2
    )
    print(f"  MC ES          : {'OK' if _verifier(TwoRoundRPS, V_star, gamma, etats, 3, Q_es) else 'ECHEC'}")

    Q_on = mc_on_policy_first_visit(
        TwoRoundRPS(rng=random.Random(GRAINE)), nb_episodes=10000, gamma=gamma, epsilon=0.1
    )
    print(f"  MC on-policy   : {'OK' if _verifier(TwoRoundRPS, V_star, gamma, etats, 3, Q_on) else 'ECHEC'}")

    _, Q_off = off_policy_mc_control(
        TwoRoundRPS(rng=random.Random(GRAINE)),
        n_episodes=10000,
        gamma=gamma,
        epsilon=0.1,
        rng=random.Random(GRAINE + 1),
    )
    print(f"  MC off-policy  : {'OK' if _verifier(TwoRoundRPS, V_star, gamma, etats, 3, Q_off) else 'ECHEC'}")
    print()


if __name__ == "__main__":
    comparer_grid_world()
    comparer_monty_hall_2()
    comparer_rock_paper_scissors()
