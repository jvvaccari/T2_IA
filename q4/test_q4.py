import unittest

import numpy as np

from q4.experiments import run_experiment
from q4.mountain_car import MountainCarEnvironment, refined_features, simple_features


class LinearMountainCarTests(unittest.TestCase):
    def test_feature_shapes(self):
        environment = MountainCarEnvironment(6, 5)
        try:
            state, _ = environment.reset()
            simple = simple_features(environment, state, 0)
            refined = refined_features(environment, state, 0)
            self.assertEqual(simple.shape, (12,))
            self.assertGreater(refined.size, simple.size)
            self.assertTrue(np.isfinite(refined).all())
        finally:
            environment.close()

    def test_short_experiment(self):
        result = run_experiment(
            feature_mode="refined",
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
