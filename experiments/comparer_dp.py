# Compare Policy Iteration et Value Iteration sur Grid World, Monty Hall 2 et
# Rock Paper Scissors : les deux algos DP doivent converger vers le meme
# V*/politique, sinon il y a un bug dans l'un des deux.
import time

from algorithms.dynamic_programming import policy_iteration, value_iteration
from environments.grid_world import GridWorld
from environments.monty_hall_2 import MontyHall2
from environments.rock_paper_scissors import TwoRoundRPS


def comparer(nom, fabrique_env, gamma) -> None:
    t0 = time.perf_counter()
    pi_pol, V_pol = policy_iteration(fabrique_env(), gamma=gamma)
    t_pol = time.perf_counter() - t0

    t0 = time.perf_counter()
    pi_val, V_val = value_iteration(fabrique_env(), gamma=gamma)
    t_val = time.perf_counter() - t0

    memes_V = all(abs(a - b) < 1e-6 for a, b in zip(V_pol, V_val))
    memes_politiques = all(
        pi_pol.meilleure_action(s) == pi_val.meilleure_action(s) for s in range(len(V_pol))
    )

    print(f"=== {nom} ===")
    print(f"  Policy Iteration : {t_pol * 1000:6.2f} ms")
    print(f"  Value Iteration  : {t_val * 1000:6.2f} ms")
    print(f"  V* identiques          : {memes_V}")
    print(f"  politiques identiques  : {memes_politiques}")
    if not memes_V or not memes_politiques:
        print("  V (Policy Iteration) :", [round(v, 3) for v in V_pol])
        print("  V (Value Iteration)  :", [round(v, 3) for v in V_val])
    print()


if __name__ == "__main__":
    comparer("Grid World", GridWorld, gamma=0.999999)
    comparer("Monty Hall 2", MontyHall2, gamma=1.0)
    comparer("Rock Paper Scissors", TwoRoundRPS, gamma=1.0)
