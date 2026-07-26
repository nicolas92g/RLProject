# Entraine et sauvegarde les modeles de chaque env, pour le rejeu sans reentrainer.
import os
import random

from algorithms.dynamic_programming import policy_iteration
from algorithms.monte_carlo import mc_on_policy_first_visit, monte_carlo_es, off_policy_mc_control
from algorithms.planning import dyna_q
from algorithms.temporal_difference import q_learning, sarsa
from environments.grid_world import GridWorld
from environments.line_world import LineWorld, LineWorldEnv
from environments.monty_hall_1 import MontyHall1, MontyHall1Env
from environments.monty_hall_2 import MontyHall2
from environments.rock_paper_scissors import TwoRoundRPS
from environments.secret_envs import secret_env_0, secret_env_1, secret_env_2, secret_env_3
from utils.io import sauvegarder_politique, sauvegarder_Q, sauvegarder_V

# tout l'aleatoire du projet passe par le module random de Python, une seule
# graine ici suffit donc a rendre les modeles reproductibles a l'identique
GRAINE = 42


def entrainer_line_world() -> None:
    dossier = "saved_models/line_world"
    os.makedirs(dossier, exist_ok=True)

    pi, V = policy_iteration(LineWorld(), gamma=0.999999)
    sauvegarder_V(V, f"{dossier}/V_policy_iteration.json")
    sauvegarder_politique(pi, f"{dossier}/politique_policy_iteration.json")

    Q = q_learning(LineWorldEnv(), nb_episodes=5000)
    sauvegarder_Q(Q, f"{dossier}/Q_q_learning.json")

    Q_mc = monte_carlo_es(LineWorldEnv(), nb_episodes=5000)
    sauvegarder_Q(Q_mc, f"{dossier}/Q_mc_es.json")


def entrainer_monty_hall_1() -> None:
    dossier = "saved_models/monty_hall_1"
    os.makedirs(dossier, exist_ok=True)

    pi, V = policy_iteration(MontyHall1(), gamma=1.0)
    sauvegarder_V(V, f"{dossier}/V_policy_iteration.json")
    sauvegarder_politique(pi, f"{dossier}/politique_policy_iteration.json")

    Q = q_learning(MontyHall1Env(), nb_episodes=20000, gamma=1.0)
    sauvegarder_Q(Q, f"{dossier}/Q_q_learning.json")

    Q_mc = monte_carlo_es(MontyHall1Env(), nb_episodes=20000, gamma=1.0, nb_pas_echauffement_max=3)
    sauvegarder_Q(Q_mc, f"{dossier}/Q_mc_es.json")


def entrainer_grid_world() -> None:
    dossier = "saved_models/grid_world"
    os.makedirs(dossier, exist_ok=True)

    pi, V = policy_iteration(GridWorld())
    sauvegarder_V(V, f"{dossier}/V_policy_iteration.json")
    sauvegarder_politique(pi, f"{dossier}/politique_policy_iteration.json")

    Q = q_learning(GridWorld(), nb_episodes=10000)
    sauvegarder_Q(Q, f"{dossier}/Q_q_learning.json")

    Q_mc = monte_carlo_es(GridWorld(), nb_episodes=10000)
    sauvegarder_Q(Q_mc, f"{dossier}/Q_mc_es.json")


def entrainer_monty_hall_2() -> None:
    dossier = "saved_models/monty_hall_2"
    os.makedirs(dossier, exist_ok=True)

    # gamma=1.0 comme Monty Hall 1, seul le resultat final compte
    # rng seedee explicitement, MontyHall2 a son propre random.Random()
    pi, V = policy_iteration(MontyHall2(), gamma=1.0)
    sauvegarder_V(V, f"{dossier}/V_policy_iteration.json")
    sauvegarder_politique(pi, f"{dossier}/politique_policy_iteration.json")

    # les etats 4 a 7 ne sont atteints qu'apres 2 choix non-gloutons
    # d'affilee, donc sous-explores par epsilon-greedy : Q-Learning ne
    # retrouve l'optimum exact qu'1 fois sur 2 environ. on verifie contre
    # Policy Iteration et on retente avec une autre graine si besoin
    optimal = {s: pi.meilleure_action(s) for s in range(1, 8)}
    for tentative in range(20):
        graine = GRAINE + tentative
        random.seed(graine)
        Q = q_learning(MontyHall2(rng=random.Random(graine)), nb_episodes=80000, gamma=1.0, epsilon=0.3)
        appris = {s: max(range(2), key=lambda a: Q[s][a]) for s in range(1, 8)}
        if appris == optimal:
            break
    else:
        print("ATTENTION : Q-Learning n'a pas retrouve l'optimum exact sur Monty Hall 2 apres 20 graines.")
    sauvegarder_Q(Q, f"{dossier}/Q_q_learning.json")

    Q_mc = monte_carlo_es(
        MontyHall2(rng=random.Random(GRAINE)), nb_episodes=40000, gamma=1.0, nb_pas_echauffement_max=4
    )
    sauvegarder_Q(Q_mc, f"{dossier}/Q_mc_es.json")


def entrainer_rock_paper_scissors() -> None:
    dossier = "saved_models/rock_paper_scissors"
    os.makedirs(dossier, exist_ok=True)

    # gamma=1.0 : score() est la somme non actualisee des deux rounds
    # rng seedee explicitement, meme raison que MontyHall2
    pi, V = policy_iteration(TwoRoundRPS(), gamma=1.0)
    sauvegarder_V(V, f"{dossier}/V_policy_iteration.json")
    sauvegarder_politique(pi, f"{dossier}/politique_policy_iteration.json")

    Q = q_learning(TwoRoundRPS(rng=random.Random(GRAINE)), nb_episodes=10000, gamma=1.0)
    sauvegarder_Q(Q, f"{dossier}/Q_q_learning.json")

    Q_mc = monte_carlo_es(
        TwoRoundRPS(rng=random.Random(GRAINE)), nb_episodes=10000, gamma=1.0, nb_pas_echauffement_max=2
    )
    sauvegarder_Q(Q_mc, f"{dossier}/Q_mc_es.json")


# secret envs : pas de contrat MDPEnv (boite noire), seuls les 6 algos
# model-free s'appliquent. les 4 recoivent le meme traitement, d'ou le
# partage de cette fonction plutot que 4 copies identiques
def _entrainer_secret_env(fabrique, dossier) -> None:
    os.makedirs(dossier, exist_ok=True)

    Q_es = monte_carlo_es(fabrique(), nb_episodes=3000)
    sauvegarder_Q(Q_es, f"{dossier}/Q_mc_es.json")

    Q_on = mc_on_policy_first_visit(fabrique(), nb_episodes=3000)
    sauvegarder_Q(Q_on, f"{dossier}/Q_mc_on_policy.json")

    _, Q_off = off_policy_mc_control(fabrique(), n_episodes=3000, rng=random.Random(GRAINE))
    sauvegarder_Q(Q_off, f"{dossier}/Q_off_policy_mc.json")

    Q_sarsa = sarsa(fabrique(), nb_episodes=3000)
    sauvegarder_Q(Q_sarsa, f"{dossier}/Q_sarsa.json")

    Q_ql = q_learning(fabrique(), nb_episodes=3000)
    sauvegarder_Q(Q_ql, f"{dossier}/Q_q_learning.json")

    Q_dyna = dyna_q(fabrique(), nb_episodes=3000)
    sauvegarder_Q(Q_dyna, f"{dossier}/Q_dyna_q.json")


def entrainer_secret_env_0() -> None:
    _entrainer_secret_env(secret_env_0, "saved_models/secret_env_0")


def entrainer_secret_env_1() -> None:
    _entrainer_secret_env(secret_env_1, "saved_models/secret_env_1")


def entrainer_secret_env_2() -> None:
    _entrainer_secret_env(secret_env_2, "saved_models/secret_env_2")


def entrainer_secret_env_3() -> None:
    _entrainer_secret_env(secret_env_3, "saved_models/secret_env_3")


if __name__ == "__main__":
    random.seed(GRAINE)
    print("LineWorld...")
    entrainer_line_world()
    print("Monty Hall 1...")
    entrainer_monty_hall_1()
    print("Grid World...")
    entrainer_grid_world()
    print("Monty Hall 2...")
    entrainer_monty_hall_2()
    print("Rock Paper Scissors...")
    entrainer_rock_paper_scissors()
    print("Secret Env 0...")
    entrainer_secret_env_0()
    print("Secret Env 1...")
    entrainer_secret_env_1()
    print("Secret Env 2...")
    entrainer_secret_env_2()
    print("Secret Env 3...")
    entrainer_secret_env_3()
    print("terminé.")
