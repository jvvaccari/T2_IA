import unittest

from q2.experiments import run_experiment
from q2.gym_adapter import make_environment


class QLearningEnvironmentTests(unittest.TestCase):
    def test_environments_expose_discrete_tables(self):
        for name, kwargs in (
            ("Blackjack-v1", {"natural": False, "sab": False}),
            ("CliffWalking-v1", {}),
            ("FrozenLake-v1", {"map_name": "4x4", "is_slippery": True}),
        ):
            with self.subTest(environment=name):
                environment = make_environment(name, **kwargs)
                try:
                    self.assertGreater(environment.get_num_states(), 0)
                    self.assertGreater(environment.get_num_actions(), 0)
                    state, _ = environment.reset()
                    state_id = environment.get_state_id(state)
                    self.assertGreaterEqual(state_id, 0)
                finally:
                    environment.close()

    def test_short_experiments(self):
        for name in ("blackjack", "cliffwalking", "frozenlake"):
            with self.subTest(environment=name):
                result = run_experiment(name, episodes=3, test_episodes=2)
                self.assertEqual(len(result["history"]["rewards"]), 3)
                self.assertGreaterEqual(result["metrics"]["success_rate"], 0.0)
                self.assertLessEqual(result["metrics"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
