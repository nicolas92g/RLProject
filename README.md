# RLProject

Implémentation pédagogique d'algorithmes de Reinforcement Learning, du cas où le
modèle de l'environnement est connu (Dynamic Programming) jusqu'à l'apprentissage
par seule interaction (model-free).

## État d'avancement

| Module | État |
|---|---|
| `environments/base.py` | ✅ contrats `MDPEnv` et `ModelFreeEnv` |
| `environments/line_world.py` | ✅ `MDPEnv` — ⬜ `ModelFreeEnv` |
| `environments/*` (grid_world, monty_hall_1/2, rock_paper_scissors) | ⬜ à implémenter |
| `algorithms/dynamic_programming.py` | ✅ Policy Iteration, Value Iteration (LineWorld) |
| `algorithms/*` (monte_carlo, temporal_difference, planning) | ⬜ à implémenter |
| `utils/policy.py` | ✅ représentation d'une politique |
| `utils/io.py` | ⬜ à implémenter |

## Les deux contrats d'environnement

Tout environnement implémente l'un des deux contrats définis dans
`environments/base.py`. Les algorithmes ne dépendent que du contrat, jamais d'un
environnement concret.

### `MDPEnv` — modèle complet connu

L'environnement expose ses probabilités de transition et ses récompenses. C'est ce
que requièrent Policy Iteration et Value Iteration.

```python
num_states() -> int
num_actions() -> int
num_rewards() -> int
reward(i) -> float                  # récompense d'indice i
p(s, a, s_p, r_index) -> float      # P(s', r | s, a)
```

### `ModelFreeEnv` — interaction seule

L'agent n'a accès qu'à `reset` / `step` et découvre la dynamique en jouant. C'est ce
que requièrent Monte Carlo, TD Learning et Planning.

```python
reset() -> None
step(action) -> None
is_game_over() -> bool
current_state() -> int
available_actions() -> List[int]
score() -> float

maximum_states_count() -> int       # pré-allocation des tableaux V/Q uniquement
maximum_actions_count() -> int
pretty_print() -> None              # optionnel, pour le débogage
```

`maximum_states_count()` et `maximum_actions_count()` servent exclusivement à
dimensionner les tableaux à l'avance — ils ne doivent pas intervenir dans la logique
d'apprentissage.

## Algorithmes visés

| Famille | Contrat requis | Algorithmes |
|---|---|---|
| Dynamic Programming | `MDPEnv` | Policy Iteration, Value Iteration |
| Monte Carlo | `ModelFreeEnv` | Monte Carlo ES, on-policy first-visit control, off-policy control |
| Temporal Difference | `ModelFreeEnv` | Sarsa, Q-Learning |
| Planning | `ModelFreeEnv` | Dyna-Q (Dyna-Q+ optionnel) |

Liste exacte du sujet (`documents/SyllabusDuProjet.pdf`) — Expected Sarsa et
Every-visit MC sont vus en cours mais **non demandés**, volontairement absents d'ici.

## Environnements visés

- **Line World** — le plus simple, sert de banc de validation (V\* calculable à la main)
- **Grid World** — extension 2D
- **Monty Hall (1 et 2)** — épisodes très courts, sensibles au facteur d'actualisation
- **Rock Paper Scissors** — adversaire stochastique

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` liste numpy, matplotlib, jupyter, ipykernel, tqdm, pygame
(versions non épinglées pour l'instant).

## Organisation du dépôt

```
algorithms/     implémentations RL
environments/   problèmes test + contrats (base.py)
utils/          représentation des politiques, entrées/sorties
experiments/    scripts d'expérimentation (résultats non versionnés)
saved_models/   checkpoints (non versionnés)
report/         rapport et visualisations
libs/           dépendances externes éventuelles
```

## Conventions

- Code et docstrings en **français**, comme `environments/base.py`
- Un algorithme ne dépend que du contrat d'environnement, pas d'une implémentation
- Messages de commit : `type: description` (`feat:`, `fix:`, `docs:`, `refactor:`)
