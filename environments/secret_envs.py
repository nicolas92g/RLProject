# Adaptateur ModelFreeEnv pour les secret envs 0 à 3 (wrapper fourni par le prof).
import os
import platform
from typing import List

from environments import secret_envs_wrapper as wrapper
from environments.base import ModelFreeEnv

# le wrapper fourni résout lib_path en relatif ("./libs/...") selon le cwd du process ;
# on le remplace par un chemin absolu pour ne pas dépendre du répertoire d'exécution
_LIBS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs")
_SYSTEM = platform.system().lower()
if _SYSTEM == "windows":
    wrapper.lib_path = os.path.join(_LIBS_DIR, "secret_envs.dll")
elif _SYSTEM == "linux":
    wrapper.lib_path = os.path.join(_LIBS_DIR, "libsecret_envs.so")
elif _SYSTEM == "darwin":
    if "intel" in platform.processor().lower():
        wrapper.lib_path = os.path.join(_LIBS_DIR, "libsecret_envs_intel_macos.dylib")
    else:
        wrapper.lib_path = os.path.join(_LIBS_DIR, "libsecret_envs.dylib")


class SecretEnvAdapter(ModelFreeEnv):
    # traduit l'interface du wrapper (state_id/display/np.uint64) vers notre contrat
    def __init__(self, secret_env_cls) -> None:
        self._env = secret_env_cls()

    def reset(self) -> None:
        self._env.reset()

    def step(self, action: int) -> None:
        self._env.step(action)

    def is_game_over(self) -> bool:
        return self._env.is_game_over()

    def current_state(self) -> int:
        return self._env.state_id()

    def available_actions(self) -> List[int]:
        # np.uint64 -> int, sinon ctypes.step() plante
        return [int(a) for a in self._env.available_actions()]

    def score(self) -> float:
        return self._env.score()

    def pretty_print(self) -> None:
        self._env.display()

    def maximum_states_count(self) -> int:
        return self._env.num_states()

    def maximum_actions_count(self) -> int:
        return self._env.num_actions()


def secret_env_0() -> SecretEnvAdapter:
    return SecretEnvAdapter(wrapper.SecretEnv0)


def secret_env_1() -> SecretEnvAdapter:
    return SecretEnvAdapter(wrapper.SecretEnv1)


def secret_env_2() -> SecretEnvAdapter:
    return SecretEnvAdapter(wrapper.SecretEnv2)


def secret_env_3() -> SecretEnvAdapter:
    return SecretEnvAdapter(wrapper.SecretEnv3)
