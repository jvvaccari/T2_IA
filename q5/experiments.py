import argparse
from typing import Any

from q2.experiments import ENVIRONMENTS
from q2.gym_adapter import make_environment
from q2.qlt import QLearningAgentTabular

from .sarsa import SARSAAgent


def evaluate(
    agent, environment, episodes: int, record_trajectory: bool = False
) -> dict[str, Any]:
    rewards: list[float] = []
    successes = 0
    trajectories: list[list[tuple[int, int]]] = []

    for _ in range(episodes):
        state, _ = environment.reset()
        state_id = environment.get_state_id(state)
        total_reward = 0.0
        trajectory: list[tuple[int, int]] = []

        for _ in range(1000):
            action = agent.choose_action(state_id, is_in_exploration_mode=False)
            if record_trajectory:
                trajectory.append((state_id, action))
            next_state, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            state_id = environment.get_state_id(next_state)
            if terminated or truncated:
                successes += int(reward > 0)
                break

        rewards.append(total_reward)
        if record_trajectory:
            trajectories.append(trajectory)

    result: dict[str, Any] = {
        "average_reward": sum(rewards) / episodes,
        "success_rate": successes / episodes,
    }
    if record_trajectory:
        result["trajectories"] = trajectories
    return result


def _make_agent(kind: str, environment, learning_rate, gamma, decay, verbose):
    agent_class = (
        QLearningAgentTabular if kind == "qlearning" else SARSAAgent
    )
    return agent_class(
        environment,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_decay_rate=decay,
        verbose=verbose,
    )


def run_experiment(
    environment_name: str,
    algorithm: str,
    episodes: int = 1000,
    test_episodes: int = 100,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon_decay_rate: float = 0.001,
    verbose: bool = False,
) -> dict[str, Any]:
    if algorithm not in ("qlearning", "sarsa"):
        raise ValueError("algorithm deve ser 'qlearning' ou 'sarsa'.")

    gym_name, kwargs = ENVIRONMENTS[environment_name]
    environment = make_environment(gym_name, **kwargs)
    try:
        agent = _make_agent(
            algorithm,
            environment,
            learning_rate,
            gamma,
            epsilon_decay_rate,
            verbose,
        )
        history = agent.train(episodes)
        metrics = evaluate(
            agent,
            environment,
            test_episodes,
            record_trajectory=environment_name == "cliffwalking",
        )
        return {"history": history, "metrics": metrics, "agent": agent}
    finally:
        environment.close()


def compare(
    environment_name: str,
    episodes: int = 1000,
    test_episodes: int = 100,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon_decay_rate: float = 0.001,
) -> dict[str, dict[str, Any]]:
    return {
        algorithm: run_experiment(
            environment_name,
            algorithm,
            episodes,
            test_episodes,
            learning_rate,
            gamma,
            epsilon_decay_rate,
        )
        for algorithm in ("qlearning", "sarsa")
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparação entre Q-learning e SARSA."
    )
    parser.add_argument("--environment", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay-rate", type=float, default=0.001)
    args = parser.parse_args()

    results = compare(
        environment_name=args.environment,
        episodes=args.episodes,
        test_episodes=args.test_episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay_rate=args.epsilon_decay_rate,
    )
    for algorithm, result in results.items():
        print(f"\n{algorithm.upper()}")
        print(f"average_reward: {result['metrics']['average_reward']:.4f}")
        print(f"success_rate: {result['metrics']['success_rate']:.4f}")
        print(
            "final_training_reward:",
            result["history"]["rewards"][-1],
        )


if __name__ == "__main__":
    main()
