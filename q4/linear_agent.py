import math
from collections.abc import Callable
from timeit import default_timer as timer
from typing import Any

import numpy as np


FeatureFunction = Callable[[Any, int], np.ndarray]


class QLearningAgentLinear:
    """Q-learning com Q(s,a) representado por pesos e features."""

    def __init__(
        self,
        env,
        feature_function: FeatureFunction,
        num_features: int,
        learning_rate: float,
        gamma: float,
        epsilon_decay_rate: float,
        min_epsilon: float = 0.01,
        max_epsilon: float = 1.0,
        verbose: bool = True,
    ):
        self.env = env
        self.feature_function = feature_function
        self.weights = np.zeros((env.get_num_actions(), num_features))
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = epsilon_decay_rate
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose = verbose
        self.history = {
            "rewards": [],
            "penalties": [],
            "steps": [],
            "epsilons": [],
            "weight_norms": [],
        }

        if self.verbose:
            print(f"Feature vector size: {num_features}")
            print(f"Weight matrix shape: {self.weights.shape}")

    def features(self, state: Any, action: int) -> np.ndarray:
        values = np.asarray(self.feature_function(state, action), dtype=float)
        if values.shape != (self.weights.shape[1],):
            raise ValueError(
                f"Features devem ter shape {(self.weights.shape[1],)}, "
                f"mas receberam {values.shape}."
            )
        return values

    def q_value(self, state: Any, action: int) -> float:
        return float(np.dot(self.weights[action], self.features(state, action)))

    def q_values(self, state: Any) -> np.ndarray:
        return np.array(
            [self.q_value(state, action) for action in range(self.env.get_num_actions())]
        )

    def choose_action(self, state: Any, is_in_exploration_mode: bool = True) -> int:
        if is_in_exploration_mode and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.env.get_num_actions()))
        return int(np.argmax(self.q_values(state)))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        terminal: bool = False,
    ) -> float:
        current = self.q_value(state, action)
        next_value = 0.0 if terminal else float(np.max(self.q_values(next_state)))
        td_error = reward + self.gamma * next_value - current
        self.weights[action] += self.learning_rate * td_error * self.features(state, action)
        return float(td_error)

    def _epsilon_decay(self, episode: int) -> None:
        self.epsilon = self.min_epsilon + (
            self.max_epsilon - self.min_epsilon
        ) * math.exp(-self.decay_rate * episode)
        self.history["epsilons"].append(self.epsilon)

    def train(self, num_episodes: int) -> dict[str, list[float]]:
        start_time = timer()
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward, penalties, steps = 0.0, 0, 0

            while True:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                terminal = terminated or truncated
                self.update(state, action, reward, next_state, terminal)
                total_reward += reward
                penalties += int(reward < 0)
                steps += 1
                state = next_state

                if terminal:
                    self._epsilon_decay(episode)
                    break

            self.history["rewards"].append(total_reward)
            self.history["penalties"].append(penalties)
            self.history["steps"].append(steps)
            self.history["weight_norms"].append(float(np.linalg.norm(self.weights)))

            if self.verbose and episode % 100 == 0:
                elapsed = timer() - start_time
                print(
                    f"Episode {episode}/{num_episodes}\n"
                    f"\tSteps: {steps}\n"
                    f"\tReward: {total_reward:.2f}\n"
                    f"\tEpsilon: {self.epsilon:.4f}\n"
                    f"\tWeight norm: {self.history['weight_norms'][-1]:.4f}\n"
                    f"\tElapsed: {elapsed:.2f}s"
                )
                start_time = timer()
        return self.history
