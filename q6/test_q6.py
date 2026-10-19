import unittest

from q6.environment import make_cliff_environment
from q6.experiments import episodes_to_convergence, run_experiment
from q6.potential import cliff_potential


class RewardShapingTests(unittest.TestCase):
    def test_potential_values(self):
        self.assertEqual(cliff_potential(0), -14.0)
        self.assertEqual(cliff_potential(47), 0.0)

    def test_shaping_reward_matches_formula(self):
        environment = make_cliff_environment(gamma=0.99, shaping=True)
        try:
            state, _ = environment.reset()
            state_id = environment.get_state_id(state)
            next_state, reward, *_ = environment.step(0)
            next_id = environment.get_state_id(next_state)
            base_reward = -1.0
            expected = base_reward + 0.99 * cliff_potential(next_id) - cliff_potential(state_id)
            self.assertEqual(reward, expected)
        finally:
            environment.close()

    def test_short_comparison(self):
        result = run_experiment(shaping=True, episodes=3, test_episodes=2)
        self.assertEqual(len(result["history"]["rewards"]), 3)
        self.assertIsNone(episodes_to_convergence([0.0] * 3))


if __name__ == "__main__":
    unittest.main()
