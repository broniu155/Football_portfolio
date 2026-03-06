import unittest

import pandas as pd

from app.components.set_pieces import (
    classify_corner_side,
    classify_free_kick_type,
    classify_target_zone,
    extract_set_piece_events,
)


class SetPiecesTests(unittest.TestCase):
    def test_rule_based_classifications(self) -> None:
        self.assertEqual(classify_corner_side(10), "Left")
        self.assertEqual(classify_corner_side(70), "Right")
        self.assertEqual(classify_target_zone(116, 40), "Six-yard central")
        self.assertEqual(classify_target_zone(110, 20), "Near-post")

        direct_fk = pd.Series({"type_name": "Shot"})
        self.assertEqual(classify_free_kick_type(direct_fk), "Direct")

    def test_extract_set_piece_events_schema_and_linkage(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "match_id": 1001,
                    "event_index": 10,
                    "team_id": 1,
                    "team_name": "North City",
                    "player_name": "Taker A",
                    "type_name": "Pass",
                    "play_pattern_name": "From Corner",
                    "period": 1,
                    "minute": 10,
                    "second": 0,
                    "location_x": 120,
                    "location_y": 5,
                    "pass_end_location_x": 111,
                    "pass_end_location_y": 38,
                    "pass_outcome_name": "",
                },
                {
                    "event_id": 2,
                    "match_id": 1001,
                    "event_index": 12,
                    "team_id": 1,
                    "team_name": "North City",
                    "player_name": "Shooter A",
                    "type_name": "Shot",
                    "play_pattern_name": "From Corner",
                    "period": 1,
                    "minute": 10,
                    "second": 8,
                    "location_x": 110,
                    "location_y": 38,
                    "shot_end_location_x": 119,
                    "shot_end_location_y": 40,
                    "shot_outcome_name": "Goal",
                },
                {
                    "event_id": 3,
                    "match_id": 1001,
                    "event_index": 20,
                    "team_id": 2,
                    "team_name": "South United",
                    "player_name": "Taker B",
                    "type_name": "Pass",
                    "play_pattern_name": "From Free Kick",
                    "period": 1,
                    "minute": 20,
                    "second": 0,
                    "location_x": 70,
                    "location_y": 60,
                    "pass_end_location_x": 74,
                    "pass_end_location_y": 58,
                    "pass_cross": False,
                    "pass_outcome_name": "Incomplete",
                },
            ]
        )

        out = extract_set_piece_events(events, follow_up_seconds=15, next_n_actions=5)
        required_cols = {
            "match_id",
            "team",
            "player",
            "minute",
            "period",
            "set_piece_type",
            "side",
            "subtype",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "target_zone",
            "outcome",
            "recipient",
            "linked_shot",
            "linked_goal",
            "short_set_piece",
        }
        self.assertTrue(required_cols.issubset(set(out.columns)))
        self.assertEqual(len(out), 3)

        first = out.iloc[0]
        self.assertEqual(first["set_piece_type"], "Corner")
        self.assertTrue(bool(first["linked_shot"]))
        self.assertTrue(bool(first["linked_goal"]))

        third = out.iloc[2]
        self.assertEqual(third["set_piece_type"], "Free Kick")
        self.assertTrue(bool(third["short_set_piece"]))


if __name__ == "__main__":
    unittest.main()
