from typing import Any

import gymnasium as gym
import numpy as np


class MountainCarEnvironment:
    def __init__(self, position_bins: int = 18, velocity_bins: int = 14):
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
        self.position_edges = np.linspace(
            self.position_low, self.position_high, position_bins + 1
        )[1:-1]
        self.velocity_edges = np.linspace(
            self.velocity_low, self.velocity_high, velocity_bins + 1
        )[1:-1]

    def get_num_actions(self) -> int:
        if not isinstance(self.environment.action_space, gym.spaces.Discrete):
            raise TypeError("As ações precisam ser discretas.")
        return int(self.environment.action_space.n)

    def reset(self):
        return self.environment.reset()

    def step(self, action: int):
        state, reward, terminated, truncated, info = self.environment.step(action)
        return state, float(reward), terminated, truncated, info

    def close(self) -> None:
        self.environment.close()

    def bin_state(self, state: Any) -> tuple[int, int]:
        position, velocity = np.asarray(state, dtype=float)
        return (
            int(np.digitize(position, self.position_edges)),
            int(np.digitize(velocity, self.velocity_edges)),
        )


def simple_features(environment: MountainCarEnvironment, state, action: int):
    position_bin, velocity_bin = environment.bin_state(state)
    values = np.zeros(
        environment.position_bins + environment.velocity_bins + 1,
        dtype=float,
    )
    values[position_bin] = 1.0
    values[environment.position_bins + velocity_bin] = 1.0
    values[-1] = action / max(environment.get_num_actions() - 1, 1)
    return values


def refined_features(environment: MountainCarEnvironment, state, action: int):
    position, velocity = np.asarray(state, dtype=float)
    position_bin, velocity_bin = environment.bin_state(state)
    position_one_hot = np.zeros(environment.position_bins)
    velocity_one_hot = np.zeros(environment.velocity_bins)
    position_one_hot[position_bin] = 1.0
    velocity_one_hot[velocity_bin] = 1.0
    action_one_hot = np.zeros(environment.get_num_actions())
    action_one_hot[action] = 1.0
    proximity = (position - environment.position_low) / (
        environment.position_high - environment.position_low
    )
    risk = float(abs(velocity) < 0.01)
    return np.concatenate(
        (
            [1.0, position, velocity, proximity, risk],
            position_one_hot,
            velocity_one_hot,
            action_one_hot,
            position_one_hot * action,
            velocity_one_hot * action,
        )
    )
