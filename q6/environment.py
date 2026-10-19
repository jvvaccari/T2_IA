from q2.gym_adapter import make_environment

from .potential import PotentialShapedEnvironment


class TrackingEnvironment:
    """Adiciona o último estado ao adaptador para calcular o potencial."""

    def __init__(self, environment, max_steps: int = 1000):
        self.environment = environment
        self.max_steps = max_steps
        self.steps = 0
        self._last_state = None

    def get_num_states(self):
        return self.environment.get_num_states()

    def get_num_actions(self):
        return self.environment.get_num_actions()

    def get_state_id(self, state):
        return self.environment.get_state_id(state)

    def reset(self):
        state, info = self.environment.reset()
        self._last_state = state
        self.steps = 0
        return state, info

    def step(self, action):
        result = self.environment.step(action)
        self._last_state = result[0]
        self.steps += 1
        state, reward, terminated, truncated, info = result
        truncated = truncated or self.steps >= self.max_steps
        return state, reward, terminated, truncated, info

    def close(self):
        self.environment.close()


def make_cliff_environment(gamma: float, shaping: bool):
    base = TrackingEnvironment(make_environment("CliffWalking-v1"))
    return PotentialShapedEnvironment(base, gamma, shaping)
