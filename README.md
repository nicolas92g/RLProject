# RLProject

Implémentation pédagogique d'algorithmes de Reinforcement Learning, du cas où le
modèle de l'environnement est connu (Dynamic Programming) jusqu'à l'apprentissage
par seule interaction (model-free).

## État d'avancement

| Module | État |
|---|---|
| `environments/base.py` | ✅ contrats `MDPEnv` et `ModelFreeEnv` |
| `environments/line_world.py` | ✅ `MDPEnv` + `ModelFreeEnv` |
| `environments/monty_hall_1.py` | ✅ `MDPEnv` + `ModelFreeEnv` |
| `environments/secret_envs.py` | ✅ `SecretEnvAdapter` + fabriques `secret_env_0()`–`secret_env_3()` |
| `environments/grid_world.py` | ✅ `MDPEnv` + `ModelFreeEnv` |
| `environments/monty_hall_2.py` | ✅ `MDPEnv` + `ModelFreeEnv` |
| `environments/rock_paper_scissors.py` | ✅ `MDPEnv` + `ModelFreeEnv` (two-round) |
| `algorithms/dynamic_programming.py` | ✅ Policy Iteration, Value Iteration |
| `algorithms/monte_carlo.py` | ✅ Monte Carlo ES, on-policy first-visit, off-policy |
| `algorithms/temporal_difference.py` | ✅ Sarsa, Q-Learning |
| `algorithms/planning.py` | ✅ Dyna-Q — ⬜ Dyna-Q+ (optionnel) |
| `utils/policy.py` | ✅ représentation d'une politique |
| `utils/io.py` | ✅ sauvegarde/chargement JSON (`V`, `Q`, `Politique`) |
| `utils/episode.py` | ✅ `run_episode()` — socle model-free commun (Δscore) |
| `utils/interaction.py` | ✅ rejeu pas-à-pas d'une politique sauvegardée + mode manuel |
| `experiments/entrainer_et_sauvegarder.py` | ✅ les 8 environnements (Policy Iteration + Q-Learning + MC ES pour les 5 `MDPEnv` ; les 6 algos model-free pour Secret Env 0-3) |
| `experiments/comparer_dp.py`, `comparer_mc.py`, `comparer_td.py`, `comparer_planning.py` | ✅ étude comparative des 4 familles d'algos vs l'optimum exact sur Grid World, Monty Hall 2, Rock Paper Scissors |
| `experiments/etudier_hyperparametres.py` | ✅ impact de α, γ, ε et n (Dyna-Q) |
| `experiments/comparer_secret_envs.py` | ✅ comparaison des 6 algos sur les 4 secret envs par score moyen rejoué |

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
pretty_print() -> None              # visualisation exigée par le sujet à chaque action
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
- **Grid World** — extension 2D (4×4, piège et objectif dans deux coins)
- **Monty Hall (1 et 2)** — épisodes très courts, sensibles au facteur d'actualisation
- **Two-Round Rock Paper Scissors** — l'adversaire rejoue au round 2 le coup joué par l'agent au round 1
- **Secret Env 0 à 3** — environnements « boîte noire » fournis par le prof
  (binaire + wrapper), accessibles via `environments/secret_envs.py`

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` liste numpy, matplotlib, jupyter, ipykernel, tqdm, pygame
(versions non épinglées pour l'instant).

## Entraîner et rejouer un modèle

```bash
python -m experiments.entrainer_et_sauvegarder   # écrit dans saved_models/
```

```python
from environments.line_world import LineWorldEnv
from utils.io import charger_Q
from utils.interaction import rejouer_politique, jouer_manuellement

Q = charger_Q("saved_models/line_world/Q_q_learning.json")
rejouer_politique(LineWorldEnv(), Q)   # rejeu pas-à-pas, sans réentraîner
jouer_manuellement(LineWorldEnv())     # mode manuel (agent humain)
```

## Organisation du dépôt

```
algorithms/     implémentations RL
environments/   problèmes test + contrats (base.py)
utils/          représentation des politiques, entrées/sorties, rejeu/mode manuel
experiments/    scripts d'expérimentation et d'entraînement
saved_models/   modèles entraînés (versionnés — livrable imposé par le sujet)
report/         rapport et visualisations
libs/           binaires des secret envs fournis par le prof (.so/.dll/.dylib)
```

## Conventions

- Identifiants (fonctions, variables, classes) en **anglais** ; docstrings et
  commentaires en **français** — voir `CLAUDE.md` pour le détail
- Un algorithme ne dépend que du contrat d'environnement, pas d'une implémentation
- Messages de commit : `type: description` (`feat:`, `fix:`, `docs:`, `refactor:`)
