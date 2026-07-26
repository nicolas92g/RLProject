# Étude de l'impact des hyperparamètres — epsilon, alpha, gamma, n pour
# Dyna-Q (TODO.md Phase 3, étape 5). Un hyperparamètre à la fois, sur
# l'environnement où son effet est le plus lisible.
#
# Métrique : erreur absolue moyenne entre Q(s,a) appris et la valeur EXACTE
# de (s,a) (calculée via Policy/Value Iteration + _valeur_action), moyennée
# sur toutes les actions des états non terminaux — pas juste "l'action
# gloutonne est-elle correcte" (trop grossier : sature vite à 0, ne montre
# pas la vitesse de convergence).
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
    print("=== epsilon (Q-Learning, Monty Hall 2, 40000 épisodes, 20 graines) ===")
    _, V_star = policy_iteration(MontyHall2(), gamma=1.0)
    etats = list(range(1, 8))
    for epsilon in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        erreurs = []
        for seed in range(20):
            random.seed(seed)
            Q = q_learning(MontyHall2(rng=random.Random(seed)), nb_episodes=40000, gamma=1.0, epsilon=epsilon)
            erreurs.append(_erreur_totale(MontyHall2, V_star, 1.0, etats, 5, Q))
        print(f"  epsilon={epsilon:<4} -> erreur Q moyenne = {sum(erreurs) / len(erreurs):.4f}")
    print("  Net décrochage sous epsilon=0.1 : les branches rares de l'arbre (états 4-7,")
    print("  cf. Phase 2) sont sous-explorées. Au-delà de ~0.3, plus de gain : le comportement")
    print("  s'éloigne trop du glouton pour que le temps d'exploration supplémentaire compense.")
    print()


def etudier_alpha() -> None:
    print("=== alpha (Q-Learning, Monty Hall 1, 2000 épisodes, 15 graines) ===")
    _, V_star = policy_iteration(MontyHall1(), gamma=1.0)
    etats = [1, 2, 3]
    for alpha in (0.01, 0.05, 0.1, 0.3, 0.6, 1.0):
        erreurs = []
        for seed in range(15):
            random.seed(seed)
            Q = q_learning(MontyHall1Env(), nb_episodes=2000, gamma=1.0, alpha=alpha)
            erreurs.append(_erreur_totale(MontyHall1, V_star, 1.0, etats, 2, Q))
        print(f"  alpha={alpha:<4} -> erreur Q moyenne = {sum(erreurs) / len(erreurs):.4f}")
    print("  Creux net autour de alpha=0.1 : en dessous, chaque transition ne corrige Q que")
    print("  d'une fraction infime (encore loin d'avoir convergé en 2000 épisodes). Au-dessus,")
    print("  chaque transition écrase la moyenne au lieu de la lisser — sur Monty Hall 1, la")
    print("  récompense finale est stochastique (gagné/perdu), donc un alpha trop grand fait")
    print("  osciller Q sur le dernier résultat tiré plutôt que de moyenner. Ne se voyait pas sur")
    print("  Line World (déterministe) : sans bruit à moyenner, alpha=1.0 y est optimal.")
    print()


def etudier_gamma() -> None:
    print("=== gamma (Value Iteration, Grid World, V*(départ)) ===")
    env = GridWorld()
    for gamma in (0.5, 0.9, 0.99, 0.999, 0.999999):
        _, V_star = value_iteration(GridWorld(), gamma=gamma)
        print(f"  gamma={gamma:<9} -> V*(départ) = {V_star[env.depart]:.4f}")
    print("  Plus gamma est petit, plus une récompense à 6 pas de distance (l'objectif) est")
    print("  dévaluée. La politique optimale ne change pas ici (un seul chemin le plus court vers")
    print("  G), mais gamma déterminerait le choix s'il existait plusieurs chemins de longueurs")
    print("  différentes vers des récompenses différentes.")
    print()


def etudier_n_planification() -> None:
    print("=== n (Dyna-Q, Grid World, 200 épisodes RÉELS seulement, 15 graines) ===")
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
    print("  n=0 équivaut à du Q-Learning pur (pas de planification) : erreur la plus haute à")
    print("  budget d'épisodes réels égal. Chute nette jusqu'à n=10, puis rendements décroissants")
    print("  (le modèle appris devient le facteur limitant, pas le nombre de relectures). Sur un")
    print("  environnement à récompense stochastique (Monty Hall 2), ce même mécanisme se")
    print("  retourne contre Dyna-Q (cf. étape 4) : rejouer plus souvent un échantillon biaisé")
    print("  aggrave le biais au lieu de le corriger.")
    print()


if __name__ == "__main__":
    etudier_epsilon()
    etudier_alpha()
    etudier_gamma()
    etudier_n_planification()
