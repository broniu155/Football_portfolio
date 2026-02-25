import unittest

import pandas as pd

from app.components.passes_metrics import attack_channel, pass_completed_mask, progressive_pass_mask


class PassesMetricsTests(unittest.TestCase):
    def test_attack_channel_boundaries(self) -> None:
        channels = attack_channel(pd.Series([10.0, 26.67, 40.0, 53.33, 70.0]))
        self.assertEqual(channels.tolist(), ["Left", "Centre", "Centre", "Centre", "Right"])

    def test_progressive_pass_classification(self) -> None:
        df = pd.DataFrame(
            {
                "location_x": [40, 70],
                "pass_end_location_x": [52, 74],
            }
        )
        mask = progressive_pass_mask(df, threshold=10.0)
        self.assertEqual(mask.tolist(), [True, False])

    def test_completion_logic_outcome_missing(self) -> None:
        df = pd.DataFrame({"pass_outcome_name": [None, "", "Incomplete"]})
        completed, available = pass_completed_mask(df)
        self.assertTrue(available)
        self.assertEqual(completed.tolist(), [True, True, False])

    def test_completion_unavailable_defaults_complete(self) -> None:
        df = pd.DataFrame({"location_x": [30, 40]})
        completed, available = pass_completed_mask(df)
        self.assertFalse(available)
        self.assertEqual(completed.tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
