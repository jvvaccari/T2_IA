from collections.abc import Hashable
from typing import Any

import gymnasium as gym
import numpy as np

from .qlt import QLearningAgentTabular


def _discrete_size(space: gym.Space) -> int:
    if not isinstance(space, gym.spaces.Discrete):
        raise TypeError("O espaço precisa ser Discrete.")
    return int(space.n)


class DiscreteGymEnvironment:
    """Adapta observações discretas do Gymnasium à interface do agente."""

    def __init__(self, environment: gym.Env):
        if not isinstance(environment.action_space, gym.spaces.Discrete):
            raise TypeError("O ambiente precisa ter um action_space discreto.")

        self.environment = environment
        self._state_ids: dict[Hashable, int] = {}

    def get_num_states(self) -> int:
        observation_space = self.environment.observation_space
        if isinstance(observation_space, gym.spaces.Discrete):
            return int(observation_space.n)
        if isinstance(observation_space, gym.spaces.Tuple):
            sizes = [_discrete_size(space) for space in observation_space.spaces]
            return int(np.prod(sizes))
        raise TypeError("A observação precisa ser Discrete ou Tuple de espaços Discrete.")

    def get_num_actions(self) -> int:
        return _discrete_size(self.environment.action_space)

    def get_state_id(self, state: Any) -> int:
        if isinstance(self.environment.observation_space, gym.spaces.Discrete):
            return int(state)

        key = self._state_key(state)
        if key not in self._state_ids:
            self._state_ids[key] = len(self._state_ids)
        return self._state_ids[key]

    def _state_key(self, state: Any) -> Hashable:
        if isinstance(state, np.ndarray):
            return tuple(state.tolist())
        if isinstance(state, (list, tuple)):
            return tuple(state)
        if isinstance(state, Hashable):
            return state
        raise TypeError(f"Estado não hashable: {state!r}")

    def reset(self):
        return self.environment.reset()

    def step(self, action: int):
        state, reward, terminated, truncated, info = self.environment.step(action)
        return state, float(reward), terminated, truncated, info

    def close(self) -> None:
        self.environment.close()


def make_environment(name: str, **kwargs: Any) -> DiscreteGymEnvironment:
    """Cria um ambiente Gymnasium e o envolve para o agente tabular."""
    try:
        environment = gym.make(name, **kwargs)
    except gym.error.VersionNotFound:
        if name != "CliffWalking-v1":
            raise
        environment = gym.make("CliffWalking-v0", **kwargs)
    return DiscreteGymEnvironment(environment)


def create_agent_environment(name: str, **kwargs: Any):
    """Retorna o adaptador e o agente com uma tabela de tamanho correto."""
    environment = make_environment(name, **kwargs)
    agent = QLearningAgentTabular(
        environment,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_decay_rate=0.001,
        verbose=False,
    )
    return environment, agent
