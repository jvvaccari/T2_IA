import argparse
from typing import Any

from .mountain_car import create_agent


def evaluate(agent, environment, episodes: int) -> dict[str, float]:
    rewards: list[float] = []
    successes = 0

    for _ in range(episodes):
        state, _ = environment.reset()
        state_id = environment.get_state_id(state)
        total_reward = 0.0

        while True:
            action = agent.choose_action(state_id, is_in_exploration_mode=False)
            next_state, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            state_id = environment.get_state_id(next_state)
            if terminated or truncated:
                if terminated and reward > -200:
                    successes += 1
                break

        rewards.append(total_reward)

    return {
        "average_reward": sum(rewards) / episodes,
        "success_rate": successes / episodes,
    }


def run_experiment(
    episodes: int = 10000,
    test_episodes: int = 100,
    position_bins: int = 18,
    velocity_bins: int = 14,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon_decay_rate: float = 0.001,
    verbose: bool = False,
) -> dict[str, Any]:
    environment, agent = create_agent(
        position_bins=position_bins,
        velocity_bins=velocity_bins,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_decay_rate=epsilon_decay_rate,
        verbose=verbose,
    )
    try:
        history = agent.train(episodes)
        metrics = evaluate(agent, environment, test_episodes)
        return {"history": history, "metrics": metrics, "agent": agent}
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-learning tabular no MountainCar discretizado."
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--position-bins", type=int, default=18)
    parser.add_argument("--velocity-bins", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay-rate", type=float, default=0.001)
    args = parser.parse_args()

    result = run_experiment(
        episodes=args.episodes,
        test_episodes=args.test_episodes,
        position_bins=args.position_bins,
        velocity_bins=args.velocity_bins,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay_rate=args.epsilon_decay_rate,
        verbose=True,
    )
    print("\nMétricas de teste:")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
