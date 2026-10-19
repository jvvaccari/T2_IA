from typing import Any

import gymnasium as gym
import numpy as np

from q2.qlt import QLearningAgentTabular


def _discrete_action_count(space: gym.Space) -> int:
    if not isinstance(space, gym.spaces.Discrete):
        raise TypeError("O espaço de ações precisa ser discreto.")
    return int(space.n)


class DiscretizedMountainCar:
    """Adapta MountainCar-v0 para estados discretos."""

    def __init__(self, position_bins: int = 18, velocity_bins: int = 14):
        if position_bins <= 0 or velocity_bins <= 0:
            raise ValueError("A quantidade de bins deve ser positiva.")

        self.environment = gym.make("MountainCar-v0")
        observation_space = self.environment.observation_space
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("O espaço de observação precisa ser Box.")
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.position_low = float(observation_space.low[0])
        self.position_high = float(observation_space.high[0])
        self.velocity_low = float(observation_space.low[1])
        self.velocity_high = float(observation_space.high[1])
        self._position_edges = np.linspace(
            self.position_low,
            self.position_high,
            position_bins + 1,
        )[1:-1]
        self._velocity_edges = np.linspace(
            self.velocity_low,
            self.velocity_high,
            velocity_bins + 1,
        )[1:-1]

    def get_num_states(self) -> int:
        return self.position_bins * self.velocity_bins

    def get_num_actions(self) -> int:
        return _discrete_action_count(self.environment.action_space)

    def get_state_id(self, state: Any) -> int:
        position, velocity = np.asarray(state, dtype=float)
        position_id = int(np.digitize(position, self._position_edges))
        velocity_id = int(np.digitize(velocity, self._velocity_edges))
        return position_id * self.velocity_bins + velocity_id

    def reset(self):
        return self.environment.reset()

    def step(self, action: int):
        state, reward, terminated, truncated, info = self.environment.step(action)
        return state, float(reward), terminated, truncated, info

    def close(self) -> None:
        self.environment.close()


def create_agent(
    position_bins: int = 18,
    velocity_bins: int = 14,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon_decay_rate: float = 0.001,
    verbose: bool = False,
) -> tuple[DiscretizedMountainCar, QLearningAgentTabular]:
    environment = DiscretizedMountainCar(position_bins, velocity_bins)
    agent = QLearningAgentTabular(
        environment,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_decay_rate=epsilon_decay_rate,
        verbose=verbose,
    )
    return environment, agent
