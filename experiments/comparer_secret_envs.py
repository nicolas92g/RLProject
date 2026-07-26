# Compare les 6 algorithmes model-free sur les 4 secret envs, a partir des
# modeles deja sauvegardes par entrainer_et_sauvegarder.py (pas de
# reentrainement ici). Pas de contrat MDPEnv pour ces boites noires, donc
# pas de V* exact contre lequel verifier : la seule mesure de qualite
# disponible est le score moyen obtenu en rejouant la politique gloutonne.
from environments.secret_envs import secret_env_0, secret_env_1, secret_env_2, secret_env_3
from utils.io import charger_Q

NB_EPISODES_EVALUATION = 200

ALGOS = (
    ("MC ES", "Q_mc_es.json"),
    ("MC on-policy", "Q_mc_on_policy.json"),
    ("MC off-policy", "Q_off_policy_mc.json"),
    ("Sarsa", "Q_sarsa.json"),
    ("Q-Learning", "Q_q_learning.json"),
    ("Dyna-Q", "Q_dyna_q.json"),
)


def _action_gloutonne(Q, s, actions_disponibles):
    meilleure = max(Q[s][a] for a in actions_disponibles)
    return next(a for a in actions_disponibles if Q[s][a] == meilleure)


def _score_moyen(fabrique, Q, nb_episodes) -> float:
    total = 0.0
    for _ in range(nb_episodes):
        env = fabrique()
        env.reset()
        while not env.is_game_over():
            a = _action_gloutonne(Q, env.current_state(), env.available_actions())
            env.step(a)
        total += env.score()
    return total / nb_episodes


def comparer(nom, fabrique, dossier) -> None:
    print(f"=== {nom} ===")
    resultats = {}
    for nom_algo, fichier in ALGOS:
        Q = charger_Q(f"{dossier}/{fichier}")
        resultats[nom_algo] = _score_moyen(fabrique, Q, NB_EPISODES_EVALUATION)
        print(f"  {nom_algo:<15}: score moyen sur {NB_EPISODES_EVALUATION} parties = {resultats[nom_algo]:+.4f}")
    meilleur = max(resultats, key=resultats.get)
    print(f"  -> meilleure stratégie trouvée : {meilleur} ({resultats[meilleur]:+.4f})")
    print()


if __name__ == "__main__":
    comparer("Secret Env 0", secret_env_0, "saved_models/secret_env_0")
    comparer("Secret Env 1", secret_env_1, "saved_models/secret_env_1")
    comparer("Secret Env 2", secret_env_2, "saved_models/secret_env_2")
    comparer("Secret Env 3", secret_env_3, "saved_models/secret_env_3")
