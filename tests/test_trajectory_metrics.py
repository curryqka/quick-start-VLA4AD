import unittest

from trajectory_utils import per_step_l2, trajectory_metrics


class TrajectoryMetricsTest(unittest.TestCase):
    def test_perfect_prediction(self):
        traj = [[float(i), 0.0, 0.0] for i in range(6)]
        metrics = trajectory_metrics(traj, traj)
        self.assertEqual(metrics["ade"], 0.0)
        self.assertEqual(metrics["fde"], 0.0)
        self.assertEqual(metrics["heading_error"], 0.0)

    def test_constant_lateral_offset(self):
        gt = [[float(i), 0.0, 0.0] for i in range(6)]
        pred = [[float(i), 2.0, 0.0] for i in range(6)]
        metrics = trajectory_metrics(pred, gt)
        self.assertAlmostEqual(metrics["ade"], 2.0)
        self.assertAlmostEqual(metrics["fde"], 2.0)
        self.assertEqual(per_step_l2(pred, gt), [2.0] * 6)

    def test_heading_error(self):
        gt = [[0.0, 0.0, 0.0] for _ in range(6)]
        pred = [[0.0, 0.0, 0.5] for _ in range(6)]
        self.assertAlmostEqual(trajectory_metrics(pred, gt)["heading_error"], 0.5)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            trajectory_metrics([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
