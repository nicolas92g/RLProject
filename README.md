# RLProject

Implémentation pédagogique d'algorithmes de Reinforcement Learning, du apprentissage avec modèle complet jusqu'au modèle-free.

## Structure

- **`algorithms/`** — Implémentations RL : Dynamic Programming, Monte Carlo, Temporal Difference, Planning
- **`environments/`** — Problèmes test : Monty Hall (1–2), Line World, Grid World, Rock Paper Scissors
- **`utils/`** — Utilitaires : gestion de politiques, I/O
- **`experiments/`** — Scripts d'expériences (résultats générés)
- **`saved_models/`** — Checkpoints et modèles entraînés
- **`report/`** — Rapports et visualisations

## Environnements

- **MDP** (modèle connu) : Policy Iteration, Value Iteration
- **Model-free** (modèle inconnu) : Monte Carlo, TD(λ), Q-Learning

## Démarrage rapide

```bash
pip install -r requirements.txt
python -m experiments.run_baseline
```

## Format des environnements

- `MDPEnv` — Contrat pour problèmes avec modèle complet
- `ModelFreeEnv` — Contrat pour apprentissage sans modèle
