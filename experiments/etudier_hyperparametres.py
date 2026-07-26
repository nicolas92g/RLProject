# Etude de l'impact des hyperparametres (epsilon, alpha, gamma, n pour
# Dyna-Q), un a la fois, sur l'environnement ou son effet est le plus lisible.
#
# Metrique : erreur absolue moyenne entre Q(s,a) appris et la valeur exacte
# de (s,a) (via Policy/Value Iteration), moyennee sur toutes les actions des
# etats non terminaux. Plus fin que "l'action gloutonne est-elle correcte",
# qui sature trop vite a 0 pour montrer la vitesse de convergence.
import random

from algorithms.dynamic_programming import _valeur_action, policy_iteration, value_iteration
from algorithms.planning import dyna_q
from algorithms.temporal_difference import q_learning
from environments.grid_world import GridWorld
from environments.monty_hall_1 import MontyHall1, MontyHall1Env
from environments.monty_hall_2 import MontyHall2


def _erreur_totale(Env, V_star, gamma, etats, nb_actions, Q) -> float:
    env = Env()
    total, n = 0.0, 0
    for s in etats:
        for a in range(nb_actions):
            total += abs(_valeur_action(env, V_star, s, a, gamma) - Q[s][a])
            n += 1
    return total / n


def etudier_epsilon() -> None:
    print("=== epsilon (Q-Learning, Monty Hall 2, 40000 episodes, 20 graines) ===")
    _, V_star = policy_iteration(MontyHall2(), gamma=1.0)
    etats = list(range(1, 8))
    for epsilon in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        erreurs = []
        for seed in range(20):
            random.seed(seed)
            Q = q_learning(MontyHall2(rng=random.Random(seed)), nb_episodes=40000, gamma=1.0, epsilon=epsilon)
            erreurs.append(_erreur_totale(MontyHall2, V_star, 1.0, etats, 5, Q))
        print(f"  epsilon={epsilon:<4} -> erreur Q moyenne = {sum(erreurs) / len(erreurs):.4f}")
    print("  decrochage net sous epsilon=0.1 : les branches rares de l'arbre sont")
    print("  sous-explorees. au-dela de ~0.3, plus de gain : le comportement s'eloigne")
    print("  trop du glouton pour que l'exploration supplementaire compense.")
    print()


def etudier_alpha() -> None:
    print("=== alpha (Q-Learning, Monty Hall 1, 2000 episodes, 15 graines) ===")
    _, V_star = policy_iteration(MontyHall1(), gamma=1.0)
    etats = [1, 2, 3]
    for alpha in (0.01, 0.05, 0.1, 0.3, 0.6, 1.0):
        erreurs = []
        for seed in range(15):
            random.seed(seed)
            Q = q_learning(MontyHall1Env(), nb_episodes=2000, gamma=1.0, alpha=alpha)
            erreurs.append(_erreur_totale(MontyHall1, V_star, 1.0, etats, 2, Q))
        print(f"  alpha={alpha:<4} -> erreur Q moyenne = {sum(erreurs) / len(erreurs):.4f}")
    print("  creux net autour de alpha=0.1 : en dessous, chaque transition ne corrige Q")
    print("  que d'une fraction infime (pas encore converge en 2000 episodes). au-dessus,")
    print("  chaque transition ecrase la moyenne au lieu de la lisser, sur Monty Hall 1 la")
    print("  recompense finale est stochastique donc un alpha trop grand fait osciller Q")
    print("  sur le dernier resultat tire. invisible sur Line World (deterministe) :")
    print("  sans bruit a moyenner, alpha=1.0 y est optimal.")
    print()


def etudier_gamma() -> None:
    print("=== gamma (Value Iteration, Grid World, V*(depart)) ===")
    env = GridWorld()
    for gamma in (0.5, 0.9, 0.99, 0.999, 0.999999):
        _, V_star = value_iteration(GridWorld(), gamma=gamma)
        print(f"  gamma={gamma:<9} -> V*(depart) = {V_star[env.depart]:.4f}")
    print("  plus gamma est petit, plus une recompense a 6 pas de distance est devaluee.")
    print("  la politique optimale ne change pas ici (chemin le plus court unique), mais")
    print("  gamma trancherait s'il existait plusieurs chemins vers des recompenses")
    print("  differentes.")
    print()


def etudier_n_planification() -> None:
    print("=== n (Dyna-Q, Grid World, 200 episodes reels seulement, 15 graines) ===")
    _, V_star = policy_iteration(GridWorld())
    env = GridWorld()
    etats = [s for s in range(env.num_states()) if s not in (env.objectif, env.piege)]
    for n_plan in (0, 1, 5, 10, 20, 50):
        erreurs = []
        for seed in range(15):
            random.seed(seed)
            Q = dyna_q(GridWorld(), nb_episodes=200, n_planification=n_plan, epsilon=0.1)
            erreurs.append(_erreur_totale(GridWorld, V_star, 0.999999, etats, 4, Q))
        print(f"  n={n_plan:<3} -> erreur Q moyenne = {sum(erreurs) / len(erreurs):.4f}")
    print("  n=0 equivaut a du Q-Learning pur : erreur la plus haute a budget d'episodes")
    print("  reels egal. chute nette jusqu'a n=10, puis rendements decroissants (le")
    print("  modele appris devient le facteur limitant). sur un environnement a")
    print("  recompense stochastique, ce meme mecanisme se retourne contre Dyna-Q :")
    print("  rejouer plus souvent un echantillon biaise aggrave le biais.")
    print()


if __name__ == "__main__":
    etudier_epsilon()
    etudier_alpha()
    etudier_gamma()
    etudier_n_planification()
