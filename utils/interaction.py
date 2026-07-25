# Rejeu pas-à-pas d'une politique apprise et mode manuel, cf. exigences transverses du
# sujet (SPECS.md §3) : visualisation à chaque action, rejeu sans réentraîner, humain.
from typing import List

from algorithms.monte_carlo import action_gloutonne
from environments.base import ModelFreeEnv


def rejouer_politique(env: ModelFreeEnv, Q: List[List[float]], pause: bool = True) -> float:
    # Q vient de utils.io.charger_Q : aucun réentraînement, seulement de la lecture
    env.reset()
    env.pretty_print()
    while not env.is_game_over():
        s = env.current_state()
        a = action_gloutonne(Q, s, env.available_actions())
        env.step(a)
        print(f"action jouée : {a}")
        env.pretty_print()
        print(f"score : {env.score()}")
        if pause:
            reponse = input("Entrée pour continuer, q pour arrêter : ")
            if reponse.strip().lower() == "q":
                break
    return env.score()


def jouer_manuellement(env: ModelFreeEnv) -> float:
    env.reset()
    env.pretty_print()
    while not env.is_game_over():
        actions_dispo = env.available_actions()
        print(f"actions possibles : {actions_dispo}")
        try:
            a = int(input("action : "))
        except ValueError:
            print("entrée invalide, entrer un numéro d'action")
            continue
        if a not in actions_dispo:
            print(f"action {a} invalide, choisir parmi {actions_dispo}")
            continue
        env.step(a)
        env.pretty_print()
        print(f"score : {env.score()}")
    print(f"partie terminée, score final : {env.score()}")
    return env.score()
