from typing import Any


class PotentialShapedEnvironment:
    """Aplica shaping potencial sem alterar a dinâmica do ambiente."""

    def __init__(self, environment, gamma: float, enabled: bool):
        self.environment = environment
        self.gamma = gamma
        self.enabled = enabled

    def get_num_states(self) -> int:
        return self.environment.get_num_states()

    def get_num_actions(self) -> int:
        return self.environment.get_num_actions()

    def get_state_id(self, state: Any) -> int:
        return self.environment.get_state_id(state)

    def reset(self):
        return self.environment.reset()

    def step(self, action: int):
        state_before = self.environment._last_state
        next_state, reward, terminated, truncated, info = self.environment.step(action)
        if self.enabled:
            current_id = self.get_state_id(state_before)
            next_id = self.get_state_id(next_state)
            reward += self.gamma * cliff_potential(next_id) - cliff_potential(current_id)
        self.environment._last_state = next_state
        return next_state, reward, terminated, truncated, info

    def close(self):
        self.environment.close()


def cliff_potential(state_id: int, columns: int = 12) -> float:
    """Potencial negativo da distância Manhattan até o objetivo (3, 11)."""
    row, column = divmod(state_id, columns)
    return -float(abs(3 - row) + abs(11 - column))
