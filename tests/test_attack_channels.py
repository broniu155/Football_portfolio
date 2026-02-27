import unittest

import pandas as pd

from app.components.attack_channels import attack_channel_from_y, compute_attack_channel, derive_attack_channel_columns


class AttackChannelsTests(unittest.TestCase):
    def test_compute_attack_channel_boundaries(self) -> None:
        left_max = 80.0 / 3.0
        right_min = 160.0 / 3.0
        self.assertEqual(compute_attack_channel(60.0, left_max).channel, "Centre")
        self.assertEqual(compute_attack_channel(60.0, right_min).channel, "Right")

    def test_compute_attack_channel_missing_and_out_of_range(self) -> None:
        missing = compute_attack_channel(None, 10.0)
        self.assertIsNone(missing.channel)
        self.assertEqual(missing.reason, "missing")

        out = compute_attack_channel(121.0, 10.0)
        self.assertIsNone(out.channel)
        self.assertEqual(out.reason, "out_of_range")

    def test_attack_channel_from_y_out_of_range_is_unknown(self) -> None:
        channels = attack_channel_from_y(pd.Series([10.0, 90.0, None]), x=pd.Series([10.0, 30.0, 40.0]))
        self.assertEqual(channels.tolist(), ["Left", "Unknown", "Unknown"])

    def test_shot_end_location_with_z_dimension(self) -> None:
        df = pd.DataFrame(
            {
                "type_name": ["Shot"],
                "location_x": [None],
                "location_y": [None],
                "shot_end_location": [[120.0, 70.0, 1.2]],
            }
        )
        out = derive_attack_channel_columns(df)
        self.assertEqual(out.iloc[0]["channel_source"], "shot.end_location")
        self.assertEqual(out.iloc[0]["channel_reason"], "ok")
        self.assertEqual(out.iloc[0]["attack_channel"], "Right")
        self.assertEqual(float(out.iloc[0]["x_used"]), 120.0)
        self.assertEqual(float(out.iloc[0]["y_used"]), 70.0)

    def test_source_fallback_and_reasons(self) -> None:
        df = pd.DataFrame(
            {
                "type_name": ["Pass", "Carry", "Shot", "Pass"],
                "location_x": [None, None, None, None],
                "location_y": [None, None, None, None],
                "pass_end_location_x": [40.0, None, None, 140.0],
                "pass_end_location_y": [10.0, None, None, 20.0],
                "carry_end_location_x": [None, 35.0, None, None],
                "carry_end_location_y": [None, 60.0, None, None],
                "shot_end_location_x": [None, None, None, None],
                "shot_end_location_y": [None, None, None, None],
                "shot_end_location": [None, None, [110.0, 30.0, 0.5], None],
            }
        )
        out = derive_attack_channel_columns(df)
        self.assertEqual(out["channel_source"].tolist(), ["pass.end_location", "carry.end_location", "shot.end_location", "pass.end_location"])
        self.assertEqual(out["channel_reason"].tolist(), ["ok", "ok", "ok", "out_of_range"])
        self.assertEqual(out["attack_channel"].tolist(), ["Left", "Right", "Centre", "Unknown"])


if __name__ == "__main__":
    unittest.main()
