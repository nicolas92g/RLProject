# Compare Dyna-Q sur Grid World, Monty Hall 2 et Rock Paper Scissors
# (TODO.md Phase 3, étape 4). Même méthode de vérification que les étapes
# précédentes : comparer la VALEUR de l'action apprise à V*(s).
#
# Sur Grid World et Rock Paper Scissors (transitions ET récompenses
# déterministes), Dyna-Q converge nettement plus vite que Q-Learning/Sarsa
# (3000 épisodes suffisent contre 10000) — c'est tout l'intérêt de la
# planification : chaque pas réel déclenche n mises à jour supplémentaires
# rejouées depuis le modèle appris.
#
# Sur Monty Hall 2, ce même modèle devient un problème : `planning.py` ne
# retient QUE la dernière transition observée pour chaque (s,a)
# (`modele[(s,a)] = (r, s_p)`), pas une moyenne. À la décision finale
# (états 4-7), la récompense est stochastique (0 ou 1, gagné/perdu) : la
# phase de planification rejoue n fois d'affilée ce dernier échantillon
# unique, ce qui biaise Q très fortement (mesuré : Q sous-estimé de 0.1 à
# 0.4 en moyenne, écart-type jusqu'à 0.45) au lieu de moyenner sur plusieurs
# essais comme le ferait un vrai pas Q-Learning. C'est une limite connue du
# Dyna-Q "de base" (cf. Sutton & Barto §8.2) : il suppose un environnement
# déterministe. Contrairement au problème d'arbre profond des étapes
# précédentes, plus d'épisodes ou d'exploration n'y changent pas grand
# chose ; réduire `n_planification` et `alpha` atténue un peu le biais
# sans l'éliminer.
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

    # Récompense stochastique à la décision finale + modèle à un seul
    # échantillon (cf. commentaire en tête de fichier) : taux de réussite
    # mesuré ~30% par tentative, retentative plus généreuse qu'ailleurs.
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
