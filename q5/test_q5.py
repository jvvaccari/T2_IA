import unittest

from q5.experiments import compare
from q5.sarsa import SARSAAgent
from q2.gym_adapter import make_environment


class SARSAAgentTests(unittest.TestCase):
    def test_sarsa_terminal_update(self):
        environment = make_environment("CliffWalking-v1")
        try:
            agent = SARSAAgent(
                environment,
                learning_rate=0.5,
                gamma=0.9,
                epsilon_decay_rate=0.1,
                verbose=False,
            )
            agent.update(0, 0, 1.0, 1, terminal=True)
            self.assertEqual(agent.q_table[0, 0], 0.5)
        finally:
            environment.close()

    def test_short_comparison(self):
        results = compare(
            "cliffwalking",
            episodes=3,
            test_episodes=2,
        )
        self.assertEqual(set(results), {"qlearning", "sarsa"})
        for result in results.values():
            self.assertEqual(len(result["history"]["rewards"]), 3)
            self.assertIn("average_reward", result["metrics"])
            self.assertIn("trajectories", result["metrics"])


if __name__ == "__main__":
    unittest.main()
