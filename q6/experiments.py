import argparse
from typing import Any, Optional

from q2.qlt import QLearningAgentTabular

from .environment import make_cliff_environment


def evaluate(agent, environment, episodes: int) -> dict[str, float]:
    rewards = []
    successes = 0
    for _ in range(episodes):
        state, _ = environment.reset()
        state_id = environment.get_state_id(state)
        total_reward = 0.0
        while True:
            action = agent.choose_action(state_id, False)
            next_state, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            state_id = environment.get_state_id(next_state)
            if terminated or truncated:
                successes += int(reward > 0)
                break
        rewards.append(total_reward)
    return {
        "average_reward": sum(rewards) / episodes,
        "success_rate": successes / episodes,
    }


def episodes_to_convergence(
    rewards: list[float],
    window: int = 100,
    threshold: float = -20.0,
) -> Optional[int]:
    for index in range(window, len(rewards) + 1):
        average = sum(rewards[index - window:index]) / window
        if average >= threshold:
            return index
    return None


def run_experiment(
    shaping: bool,
    episodes: int = 1000,
    test_episodes: int = 100,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon_decay_rate: float = 0.001,
    verbose: bool = False,
) -> dict[str, Any]:
    environment = make_cliff_environment(gamma, shaping)
    try:
        agent = QLearningAgentTabular(
            environment,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_decay_rate=epsilon_decay_rate,
            verbose=verbose,
        )
        history = agent.train(episodes)
        return {
            "history": history,
            "metrics": evaluate(agent, environment, test_episodes),
            "episodes_to_convergence": episodes_to_convergence(
                history["rewards"]
            ),
            "agent": agent,
        }
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(description="Reward shaping no Cliff Walking.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay-rate", type=float, default=0.001)
    args = parser.parse_args()

    for shaping in (False, True):
        result = run_experiment(shaping=shaping, **vars(args))
        label = "com shaping" if shaping else "sem shaping"
        print(f"\n{label.upper()}")
        print(f"average_reward: {result['metrics']['average_reward']:.4f}")
        print(f"success_rate: {result['metrics']['success_rate']:.4f}")
        print(f"episodes_to_convergence: {result['episodes_to_convergence']}")


if __name__ == "__main__":
    main()
