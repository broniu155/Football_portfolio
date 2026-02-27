import unittest

import pandas as pd

from app.components.export import build_match_events_export_df


class ExportAttackChannelTests(unittest.TestCase):
    def test_export_adds_channel_debug_columns(self) -> None:
        df = pd.DataFrame(
            {
                "type_name": ["Pass", "Shot"],
                "location_x": [20.0, None],
                "location_y": [10.0, None],
                "pass_end_location_x": [40.0, None],
                "pass_end_location_y": [12.0, None],
                "shot_end_location": [None, [120.0, 70.0, 0.8]],
                "shot_outcome_name": ["", "Goal"],
            }
        )
        out = build_match_events_export_df(df, include_derived=True, essential_only=False)
        self.assertTrue({"attack_channel", "channel_source", "channel_reason", "x_used", "y_used"}.issubset(out.columns))
        self.assertEqual(out["attack_channel"].tolist(), ["Left", "Right"])
        self.assertEqual(out["channel_source"].tolist(), ["event.location", "shot.end_location"])
        self.assertEqual(out["channel_reason"].tolist(), ["ok", "ok"])


if __name__ == "__main__":
    unittest.main()
