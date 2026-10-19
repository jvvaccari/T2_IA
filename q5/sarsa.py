import numpy as np
from typing import Optional

from q2.qlt import QLearningAgentTabular


class SARSAAgent(QLearningAgentTabular):
    """Agente SARSA on-policy baseado na implementação tabular fornecida."""

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: Optional[int] = None,
        terminal: bool = False,
    ) -> None:
        if terminal:
            target = reward
        elif next_action is None:
            raise ValueError("SARSA precisa da próxima ação para atualizar.")
        else:
            target = reward + self.gamma * self.q_table[next_state, next_action]

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def _run_episode(self, episode: int):
        state, _ = self.env.reset()
        state = self.env.get_state_id(state)
        action = self.choose_action(state)
        total_reward, penalties, steps = 0.0, 0, 0

        while True:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self.env.get_state_id(next_state)
            terminal = terminated or truncated
            next_action = None if terminal else self.choose_action(next_state)
            self.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                terminal,
            )

            total_reward += reward
            penalties += int(reward < 0)
            steps += 1
            if terminal:
                self._epsilon_decay(episode)
                break

            if next_action is None:
                raise RuntimeError("A próxima ação não foi definida.")
            state, action = next_state, next_action

        return total_reward, penalties, steps


def greedy_action(agent, state: int) -> int:
    return int(np.argmax(agent.q_table[state, :]))
