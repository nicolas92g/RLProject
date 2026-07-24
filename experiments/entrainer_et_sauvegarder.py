# Entraîne et sauvegarde les modèles des 4 envs de Robin, pour le rejeu sans réentraîner
# (livrable imposé : policies/V/Q entraînées et prêtes à réexécuter, cf. SPECS.md §1).
import os

from algorithms.dynamic_programming import policy_iteration
from algorithms.monte_carlo import monte_carlo_es
from algorithms.temporal_difference import q_learning
from environments.line_world import LineWorld, LineWorldEnv
from environments.monty_hall_1 import MontyHall1, MontyHall1Env
from environments.secret_envs import secret_env_0, secret_env_1
from utils.io import sauvegarder_politique, sauvegarder_Q, sauvegarder_V


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


def entrainer_secret_env_0() -> None:
    dossier = "saved_models/secret_env_0"
    os.makedirs(dossier, exist_ok=True)
    Q = q_learning(secret_env_0(), nb_episodes=3000)
    sauvegarder_Q(Q, f"{dossier}/Q_q_learning.json")


def entrainer_secret_env_1() -> None:
    dossier = "saved_models/secret_env_1"
    os.makedirs(dossier, exist_ok=True)
    Q = q_learning(secret_env_1(), nb_episodes=3000)
    sauvegarder_Q(Q, f"{dossier}/Q_q_learning.json")


if __name__ == "__main__":
    print("LineWorld...")
    entrainer_line_world()
    print("Monty Hall 1...")
    entrainer_monty_hall_1()
    print("Secret Env 0...")
    entrainer_secret_env_0()
    print("Secret Env 1...")
    entrainer_secret_env_1()
    print("terminé.")
