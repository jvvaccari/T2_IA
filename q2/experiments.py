import argparse
from typing import Any

from .gym_adapter import make_environment
from .qlt import QLearningAgentTabular


ENVIRONMENTS: dict[str, tuple[str, dict[str, Any]]] = {
    "blackjack": ("Blackjack-v1", {"natural": False, "sab": False}),
    "cliffwalking": ("CliffWalking-v1", {}),
    "frozenlake": (
        "FrozenLake-v1",
        {"map_name": "4x4", "is_slippery": True},
    ),
}


def evaluate(
    agent: QLearningAgentTabular,
    environment,
    episodes: int,
    max_steps: int = 1000,
) -> dict[str, float]:
    rewards = []
    successes = 0

    for _ in range(episodes):
        state, _ = environment.reset()
        state = environment.get_state_id(state)
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.choose_action(state, is_in_exploration_mode=False)
            next_state, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            state = environment.get_state_id(next_state)
            if terminated or truncated:
                if reward > 0:
                    successes += 1
                break
        rewards.append(total_reward)

    return {
        "average_reward": sum(rewards) / episodes,
        "success_rate": successes / episodes,
    }


def run_experiment(
    environment_name: str,
    episodes: int = 1000,
    test_episodes: int = 100,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon_decay_rate: float = 0.001,
    verbose: bool = False,
) -> dict[str, Any]:
    gym_name, kwargs = ENVIRONMENTS[environment_name]
    environment = make_environment(gym_name, **kwargs)
    try:
        agent = QLearningAgentTabular(
            environment,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_decay_rate=epsilon_decay_rate,
            verbose=verbose,
        )
        history = agent.train(episodes)
        metrics = evaluate(agent, environment, test_episodes)
        return {"history": history, "metrics": metrics, "agent": agent}
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino de Q-Learning tabular.")
    parser.add_argument("--environment", choices=ENVIRONMENTS, required=True)
    
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay-rate", type=float, default=0.001)
    args = parser.parse_args()

    result = run_experiment(
        environment_name=args.environment,
        episodes=args.episodes,
        test_episodes=args.test_episodes,
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
