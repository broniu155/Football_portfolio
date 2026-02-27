import unittest

import pandas as pd

from app.components.export import build_match_events_export_df


class ExportAttackChannelTests(unittest.TestCase):
    def test_export_adds_channel_debug_columns(self) -> None:
        df = pd.DataFrame(
            {
                "event_id": ["e1", "e2", "e3"],
                "match_id": [1, 1, 1],
                "type_name": ["Pass", "Shot", "Dribble"],
                "location_x": [20.0, None, 70.0],
                "location_y": [10.0, None, 50.0],
                "pass_end_location_x": [40.0, None, None],
                "pass_end_location_y": [12.0, None, None],
                "shot_end_location": [None, [120.0, 70.0, 0.8], None],
                "shot_outcome_name": ["", "Goal", ""],
                "dribble_outcome_name": [None, None, "Incomplete"],
            }
        )
        out = build_match_events_export_df(df, include_derived=True, essential_only=False)
        self.assertTrue({"attack_channel", "channel_source", "channel_reason", "x_used", "y_used"}.issubset(out.columns))
        self.assertTrue({"analysis_group", "analysis_subgroup"}.issubset(out.columns))
        self.assertTrue({"dribble_is_attempt", "dribble_is_complete", "dribble_is_incomplete", "dribble_outcome_name", "dribble_outcome_raw"}.issubset(out.columns))
        self.assertEqual(out["attack_channel"].tolist(), ["Left", "Right", "Centre"])
        self.assertEqual(out["channel_source"].tolist(), ["event.location", "shot.end_location", "event.location"])
        self.assertEqual(out["channel_reason"].tolist(), ["ok", "ok", "ok"])
        self.assertEqual(out["dribble_is_attempt"].tolist(), [False, False, True])
        self.assertEqual(out["dribble_is_incomplete"].tolist(), [False, False, True])
        self.assertEqual(out["analysis_group"].tolist(), ["passes", "offensive", "offensive"])
        self.assertEqual(out["analysis_subgroup"].tolist(), ["passes", "shots", "duels_offensive"])


if __name__ == "__main__":
    unittest.main()
