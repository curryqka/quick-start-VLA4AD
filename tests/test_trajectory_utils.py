import unittest

from trajectory_utils import (
    format_trajectory_text,
    normalize_trajectory,
    trajectory_statistics,
)

TRAJ = [[float(i), float(i) * 0.5, 0.01 * i] for i in range(1, 7)]


class TrajectoryUtilsTest(unittest.TestCase):
    def test_normalize_2d(self):
        self.assertEqual(normalize_trajectory(TRAJ), TRAJ)

    def test_normalize_3d(self):
        self.assertEqual(normalize_trajectory([TRAJ]), TRAJ)

    def test_normalize_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            normalize_trajectory([[0.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            normalize_trajectory([[0.0, 0.0] for _ in range(6)])

    def test_format_compact(self):
        text = format_trajectory_text(TRAJ, round_decimals=1)
        self.assertTrue(text.startswith("[1.0,0.5,0.0]"))
        self.assertEqual(text.count(";"), 5)

    def test_format_detailed(self):
        text = format_trajectory_text(TRAJ, format_type="detailed")
        self.assertIn("t=2.5s", text)
        self.assertIn("heading=", text)

    def test_statistics(self):
        stats = trajectory_statistics(TRAJ)
        self.assertEqual(stats["num_points"], 6)
        self.assertAlmostEqual(stats["heading_change"], 0.05)
        self.assertGreater(stats["total_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
