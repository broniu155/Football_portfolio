import unittest

import pandas as pd

from app.components.set_piece_data import extract_set_piece_events


class SetPieceDataEdgeCaseTests(unittest.TestCase):
    def test_no_corners_only_free_kicks(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": "a1",
                    "match_id": 1,
                    "event_index": 1,
                    "team_id": 10,
                    "team_name": "Team A",
                    "player_id": 101,
                    "player_name": "FK Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Free Kick",
                    "minute": 4,
                    "second": 0,
                    "period": 1,
                    "location_x": 50,
                    "location_y": 30,
                    "pass_end_location_x": 65,
                    "pass_end_location_y": 32,
                }
            ]
        )
        out = extract_set_piece_events(events)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["set_piece_type"], "Free Kick")

    def test_missing_end_coordinates_fallback_to_start(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": "a2",
                    "match_id": 1,
                    "event_index": 2,
                    "team_id": 10,
                    "team_name": "Team A",
                    "player_name": "Corner Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Corner",
                    "minute": 10,
                    "second": 5,
                    "period": 1,
                    "location_x": 120,
                    "location_y": 5,
                }
            ]
        )
        out = extract_set_piece_events(events)
        self.assertEqual(float(out.iloc[0]["start_x"]), float(out.iloc[0]["end_x"]))
        self.assertEqual(float(out.iloc[0]["start_y"]), float(out.iloc[0]["end_y"]))

    def test_missing_player_name_defaults_unknown(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": "a3",
                    "match_id": 1,
                    "event_index": 3,
                    "team_id": 10,
                    "team_name": "Team A",
                    "type_name": "Pass",
                    "play_pattern_name": "From Corner",
                    "minute": 11,
                    "second": 0,
                    "period": 1,
                    "location_x": 120,
                    "location_y": 70,
                    "pass_end_location_x": 112,
                    "pass_end_location_y": 36,
                }
            ]
        )
        out = extract_set_piece_events(events)
        self.assertEqual(str(out.iloc[0]["player"]), "Unknown")
        self.assertEqual(str(out.iloc[0]["taker"]), "Unknown")

    def test_single_event_only_data_has_no_linked_outcomes(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": "a4",
                    "match_id": 1,
                    "event_index": 4,
                    "team_id": 10,
                    "team_name": "Team A",
                    "player_name": "Corner Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Corner",
                    "minute": 12,
                    "second": 0,
                    "period": 1,
                    "location_x": 120,
                    "location_y": 5,
                    "pass_end_location_x": 111,
                    "pass_end_location_y": 39,
                }
            ]
        )
        out = extract_set_piece_events(events, include_follow_up=True)
        self.assertFalse(bool(out.iloc[0]["linked_shot"]))
        self.assertFalse(bool(out.iloc[0]["linked_goal"]))

    def test_duplicate_and_malformed_rows_do_not_crash(self) -> None:
        base = {
            "event_id": "a5",
            "match_id": 1,
            "event_index": 5,
            "team_id": 10,
            "team_name": "Team A",
            "player_name": "FK Taker",
            "type_name": "Pass",
            "play_pattern_name": "From Free Kick",
            "minute": 13,
            "second": 0,
            "period": 1,
            "location_x": 55,
            "location_y": 60,
            "pass_end_location_x": 62,
            "pass_end_location_y": 55,
        }
        malformed = {"event_id": "bad", "match_id": 1, "play_pattern_name": "From Free Kick"}
        events = pd.DataFrame([base, base.copy(), malformed])
        out = extract_set_piece_events(events)
        self.assertGreaterEqual(len(out), 2)
        self.assertTrue((out["set_piece_type"] == "Free Kick").any())

    def test_no_free_kicks_in_attacking_zone(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": "a6",
                    "match_id": 1,
                    "event_index": 6,
                    "team_id": 10,
                    "team_name": "Team A",
                    "player_name": "FK Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Free Kick",
                    "minute": 14,
                    "second": 0,
                    "period": 1,
                    "location_x": 40,
                    "location_y": 22,
                    "pass_end_location_x": 52,
                    "pass_end_location_y": 26,
                }
            ]
        )
        out = extract_set_piece_events(events)
        fk = out[out["set_piece_type"] == "Free Kick"].copy()
        fk["start_x"] = pd.to_numeric(fk["start_x"], errors="coerce")
        self.assertTrue((fk["start_x"] < 80).all())

    def test_include_follow_up_false_keeps_flags_false(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": "a7",
                    "match_id": 1,
                    "event_index": 7,
                    "team_id": 10,
                    "team_name": "Team A",
                    "player_name": "Corner Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Corner",
                    "minute": 15,
                    "second": 0,
                    "period": 1,
                    "location_x": 120,
                    "location_y": 5,
                    "pass_end_location_x": 111,
                    "pass_end_location_y": 36,
                },
                {
                    "event_id": "a8",
                    "match_id": 1,
                    "event_index": 8,
                    "team_id": 10,
                    "team_name": "Team A",
                    "player_name": "Shooter",
                    "type_name": "Shot",
                    "play_pattern_name": "From Corner",
                    "minute": 15,
                    "second": 7,
                    "period": 1,
                    "location_x": 110,
                    "location_y": 37,
                    "shot_end_location_x": 119,
                    "shot_end_location_y": 40,
                    "shot_outcome_name": "Goal",
                },
            ]
        )
        out = extract_set_piece_events(events, include_follow_up=False)
        self.assertTrue((out["linked_shot"] == False).all())  # noqa: E712
        self.assertTrue((out["linked_goal"] == False).all())  # noqa: E712


if __name__ == "__main__":
    unittest.main()
