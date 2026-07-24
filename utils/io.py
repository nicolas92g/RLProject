# Sauvegarde/chargement de V, Q et Politique en JSON (lisible, pas de dépendance).
import json
from typing import List

from utils.policy import Politique


def sauvegarder_V(V: List[float], chemin: str) -> None:
    with open(chemin, "w") as f:
        json.dump(V, f)


def charger_V(chemin: str) -> List[float]:
    with open(chemin) as f:
        return json.load(f)


def sauvegarder_Q(Q: List[List[float]], chemin: str) -> None:
    with open(chemin, "w") as f:
        json.dump(Q, f)


def charger_Q(chemin: str) -> List[List[float]]:
    with open(chemin) as f:
        return json.load(f)


def sauvegarder_politique(pi: Politique, chemin: str) -> None:
    with open(chemin, "w") as f:
        json.dump(pi._probs, f)


def charger_politique(chemin: str) -> Politique:
    with open(chemin) as f:
        probs = json.load(f)
    return Politique(probs)
