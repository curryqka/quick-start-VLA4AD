import unittest

from tools.validate_dataset import validate_record

TRAJ = [[float(i), 0.0, 0.0] for i in range(6)]


class ValidateDatasetTest(unittest.TestCase):
    def test_valid_record(self):
        record = {
            "messages": [
                {"role": "user", "content": "<image>Plan."},
                {"role": "assistant", "content": "Go straight."},
            ],
            "trajectory": TRAJ,
        }
        self.assertEqual(
            validate_record(record, check_images=False, require_trajectory=True),
            [],
        )

    def test_missing_roles(self):
        errors = validate_record(
            {"messages": [], "trajectory": TRAJ},
            check_images=False,
            require_trajectory=True,
        )
        self.assertTrue(any("messages" in error for error in errors))

    def test_missing_trajectory(self):
        record = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        errors = validate_record(
            record, check_images=False, require_trajectory=True
        )
        self.assertIn("missing top-level trajectory", errors)

    def test_bad_trajectory(self):
        record = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            "trajectory": [[0.0, 0.0, 0.0]],
        }
        errors = validate_record(
            record, check_images=False, require_trajectory=True
        )
        self.assertTrue(any("invalid trajectory" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
