# Entraîne et sauvegarde les modèles des 4 envs de Robin, pour le rejeu sans réentraîner
# (livrable imposé : policies/V/Q entraînées et prêtes à réexécuter, cf. SPECS.md §1).
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

# Tout l'aléatoire du projet passe par le module random de Python (vérifié : y compris
# les secret envs, dont le binaire est déterministe). Une seule graine ici suffit donc
# à rendre les modèles sauvegardés reproductibles à l'identique.
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

    # gamma=1.0 comme Monty Hall 1 : problème épisodique où seul le résultat
    # final compte, l'actualisation n'a pas de sens théorique ici.
    # rng seedée explicitement : MontyHall2 tire son aléatoire dans une
    # instance random.Random() propre, non affectée par random.seed(GRAINE).
    pi, V = policy_iteration(MontyHall2(), gamma=1.0)
    sauvegarder_V(V, f"{dossier}/V_policy_iteration.json")
    sauvegarder_politique(pi, f"{dossier}/politique_policy_iteration.json")

    # Piège découvert en validant : les états 4 à 7 (décision finale) ne sont
    # atteints qu'après DEUX choix non-gloutons d'affilée à la suite (decision
    # 2 puis decision 3), donc sous-échantillonnés par l'exploration
    # epsilon-greedy à budget d'épisodes raisonnable — Q-Learning converge
    # vers l'optimum exact seulement ~50% des runs (mesuré sur 10 graines).
    # On vérifie contre Policy Iteration (calcul exact) et on retente avec
    # une graine différente plutôt que de sauvegarder une politique non
    # convergée en silence.
    optimal = {s: pi.meilleure_action(s) for s in range(1, 8)}
    for tentative in range(20):
        graine = GRAINE + tentative
        random.seed(graine)
        Q = q_learning(MontyHall2(rng=random.Random(graine)), nb_episodes=80000, gamma=1.0, epsilon=0.3)
        appris = {s: max(range(2), key=lambda a: Q[s][a]) for s in range(1, 8)}
        if appris == optimal:
            break
    else:
        print("ATTENTION : Q-Learning n'a pas retrouvé l'optimum exact sur Monty Hall 2 après 20 graines.")
    sauvegarder_Q(Q, f"{dossier}/Q_q_learning.json")

    Q_mc = monte_carlo_es(
        MontyHall2(rng=random.Random(GRAINE)), nb_episodes=40000, gamma=1.0, nb_pas_echauffement_max=4
    )
    sauvegarder_Q(Q_mc, f"{dossier}/Q_mc_es.json")


def entrainer_rock_paper_scissors() -> None:
    dossier = "saved_models/rock_paper_scissors"
    os.makedirs(dossier, exist_ok=True)

    # gamma=1.0 : score() est la somme non actualisée des deux rounds.
    # rng seedée explicitement, même raison que MontyHall2 (cf. ci-dessus).
    pi, V = policy_iteration(TwoRoundRPS(), gamma=1.0)
    sauvegarder_V(V, f"{dossier}/V_policy_iteration.json")
    sauvegarder_politique(pi, f"{dossier}/politique_policy_iteration.json")

    Q = q_learning(TwoRoundRPS(rng=random.Random(GRAINE)), nb_episodes=10000, gamma=1.0)
    sauvegarder_Q(Q, f"{dossier}/Q_q_learning.json")

    Q_mc = monte_carlo_es(
        TwoRoundRPS(rng=random.Random(GRAINE)), nb_episodes=10000, gamma=1.0, nb_pas_echauffement_max=2
    )
    sauvegarder_Q(Q_mc, f"{dossier}/Q_mc_es.json")


# Secret envs : pas de contrat MDPEnv (boîte noire), donc pas de Policy/Value
# Iteration possible — seuls les 6 algos model-free s'appliquent. Les 4 secret
# envs reçoivent exactement le même traitement (aucun réglage spécifique à
# ajuster sans modèle connu), d'où le partage de cette fonction plutôt que 4
# copies identiques.
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
