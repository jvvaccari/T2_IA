import argparse
from typing import Any

from .linear_agent import QLearningAgentLinear
from .mountain_car import (
    MountainCarEnvironment,
    refined_features,
    simple_features,
)


def create_agent(feature_mode: str, **kwargs):
    environment = MountainCarEnvironment(
        kwargs.pop("position_bins", 18),
        kwargs.pop("velocity_bins", 14),
    )
    if feature_mode == "simple":
        feature_function = lambda state, action: simple_features(
            environment, state, action
        )
        num_features = environment.position_bins + environment.velocity_bins + 1
    elif feature_mode == "refined":
        feature_function = lambda state, action: refined_features(
            environment, state, action
        )
        num_features = len(feature_function([environment.position_low, 0.0], 0))
    else:
        raise ValueError("feature_mode deve ser 'simple' ou 'refined'.")

    return environment, QLearningAgentLinear(
        environment,
        feature_function,
        num_features,
        **kwargs,
    )


def evaluate(agent, environment, episodes: int) -> dict[str, float]:
    rewards = []
    successes = 0
    for _ in range(episodes):
        state, _ = environment.reset()
        total_reward = 0.0
        while True:
            action = agent.choose_action(state, False)
            state, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            if terminated or truncated:
                successes += int(terminated and reward > -200)
                break
        rewards.append(total_reward)
    return {
        "average_reward": sum(rewards) / episodes,
        "success_rate": successes / episodes,
    }


def run_experiment(
    feature_mode: str = "simple",
    episodes: int = 10000,
    test_episodes: int = 100,
    position_bins: int = 18,
    velocity_bins: int = 14,
    learning_rate: float = 0.01,
    gamma: float = 0.99,
    epsilon_decay_rate: float = 0.001,
    verbose: bool = False,
) -> dict[str, Any]:
    environment, agent = create_agent(
        feature_mode,
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


def main():
    parser = argparse.ArgumentParser(description="Q-learning linear no MountainCar.")
    parser.add_argument("--feature-mode", choices=("simple", "refined"), default="simple")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--position-bins", type=int, default=18)
    parser.add_argument("--velocity-bins", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay-rate", type=float, default=0.001)
    args = parser.parse_args()
    result = run_experiment(**vars(args), verbose=True)
    print("\nMétricas de teste:")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")
    print(f"Peso máximo absoluto: {abs(result['agent'].weights).max():.4f}")


if __name__ == "__main__":
    main()
