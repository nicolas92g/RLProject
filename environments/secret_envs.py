"""
Adaptateurs pour Secret Env 2 et Secret Env 3 (cf. documents/cours/secret_envs_wrapper.py).
Le wrapper du prof n'est jamais utilisé directement par les algos : on l'enveloppe
ici pour exposer nos contrats MDPEnv/ModelFreeEnv (environments/base.py).

Renommages : state_id() -> current_state(), display() -> pretty_print().
Env 2 et 3 partagent une API C identique (juste le numéro dans le nom des
fonctions change) : une classe de base paramétrée par _wrapper_cls évite de
recopier l'adaptateur pour zéro différence conceptuelle entre les deux.
"""

from typing import List, Optional

from documents.cours import secret_envs_wrapper as _wrapper
from environments.base import MDPEnv, ModelFreeEnv


class _SecretEnvAdapter(MDPEnv, ModelFreeEnv):
    _wrapper_cls = None  # défini par chaque sous-classe (SecretEnvNWrapper.SecretEnvN)

    def __init__(self, _env: Optional[object] = None) -> None:
        self._env = _env if _env is not None else self._wrapper_cls()

    @classmethod
    def from_random_state(cls) -> "_SecretEnvAdapter":
        return cls(_env=cls._wrapper_cls.from_random_state())

    # Contrat MDPEnv

    def num_states(self) -> int:
        return self._env.num_states()

    def num_actions(self) -> int:
        return self._env.num_actions()

    def num_rewards(self) -> int:
        return self._env.num_rewards()

    def reward(self, i: int) -> float:
        return self._env.reward(i)

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self._env.p(s, a, s_p, r_index)

    # Contrat ModelFreeEnv

    def reset(self) -> None:
        self._env.reset()

    def step(self, action: int) -> None:
        if self.is_game_over():
            raise ValueError("Épisode terminé : appeler reset() avant step().")
        self._env.step(action)

    def is_game_over(self) -> bool:
        return self._env.is_game_over()

    def current_state(self) -> int:
        return self._env.state_id()

    def available_actions(self) -> List[int]:
        return [int(a) for a in self._env.available_actions()]

    def score(self) -> float:
        return self._env.score()

    def maximum_states_count(self) -> int:
        return self.num_states()

    def maximum_actions_count(self) -> int:
        return self.num_actions()

    def pretty_print(self) -> None:
        self._env.display()

    # Extra fourni par le wrapper, utile hors contrat (debug / mode humain)
    def is_forbidden(self, action: int) -> bool:
        return bool(self._env.is_forbidden(action))


class SecretEnv2(_SecretEnvAdapter):
    _wrapper_cls = _wrapper.SecretEnv2


class SecretEnv3(_SecretEnvAdapter):
    _wrapper_cls = _wrapper.SecretEnv3
