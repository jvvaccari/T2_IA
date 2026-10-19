import unittest

from q3.experiments import run_experiment
from q3.mountain_car import DiscretizedMountainCar


class MountainCarTests(unittest.TestCase):
    def test_state_discretization(self):
        environment = DiscretizedMountainCar(
            position_bins=18,
            velocity_bins=14,
        )
        try:
            self.assertEqual(environment.get_num_states(), 18 * 14)
            self.assertEqual(environment.get_num_actions(), 3)
            state, _ = environment.reset()
            state_id = environment.get_state_id(state)
            self.assertGreaterEqual(state_id, 0)
            self.assertLess(state_id, environment.get_num_states())
        finally:
            environment.close()

    def test_short_experiment(self):
        result = run_experiment(
            episodes=3,
            test_episodes=2,
            position_bins=6,
            velocity_bins=5,
        )
        self.assertEqual(len(result["history"]["rewards"]), 3)
        self.assertGreaterEqual(result["metrics"]["success_rate"], 0.0)
        self.assertLessEqual(result["metrics"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
