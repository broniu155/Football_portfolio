import unittest

import pandas as pd

from app.components.set_piece_data import (
    SET_PIECE_OUTPUT_COLUMNS,
    classify_corner_side,
    classify_clearance_exit_lane,
    classify_delivery_subtype,
    classify_free_kick_type,
    classify_target_zone,
    compute_set_piece_sanity_checks,
    extract_defensive_corner_clearances,
    extract_set_piece_events,
    summarize_defensive_corner_clearances,
)


class SetPieceDataTests(unittest.TestCase):
    def test_classification_rules(self) -> None:
        self.assertEqual(classify_corner_side(12), "Left")
        self.assertEqual(classify_corner_side(74), "Right")
        self.assertEqual(classify_clearance_exit_lane(12), "Left")
        self.assertEqual(classify_clearance_exit_lane(40), "Centre")
        self.assertEqual(classify_clearance_exit_lane(70), "Right")
        self.assertEqual(classify_target_zone(116, 40), "Six-yard central")
        self.assertEqual(classify_target_zone(110, 22), "Near-post")
        self.assertEqual(classify_target_zone(110, 62), "Far-post")

        self.assertEqual(classify_free_kick_type(pd.Series({"type_name": "Shot"})), "Direct")
        self.assertEqual(classify_free_kick_type(pd.Series({"type_name": "Pass", "pass_cross": True})), "Crossed")

        self.assertEqual(
            classify_delivery_subtype(pd.Series({"set_piece_type": "Corner", "target_zone": "Near-post", "short_set_piece": False})),
            "Post delivery",
        )

    def test_extract_schema_and_follow_up_linkage(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "match_id": 999,
                    "event_index": 10,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 11,
                    "player_name": "Corner Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Corner",
                    "period": 1,
                    "minute": 10,
                    "second": 0,
                    "location_x": 120,
                    "location_y": 4,
                    "pass_end_location_x": 111,
                    "pass_end_location_y": 38,
                    "pass_outcome_name": "",
                    "pass_recipient_name": "Target",
                },
                {
                    "event_id": 2,
                    "match_id": 999,
                    "event_index": 12,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 12,
                    "player_name": "Shooter",
                    "type_name": "Shot",
                    "play_pattern_name": "From Corner",
                    "period": 1,
                    "minute": 10,
                    "second": 8,
                    "location_x": 109,
                    "location_y": 40,
                    "shot_end_location_x": 119,
                    "shot_end_location_y": 40,
                    "shot_outcome_name": "Goal",
                },
                {
                    "event_id": 3,
                    "match_id": 999,
                    "event_index": 22,
                    "team_id": 2,
                    "team_name": "Beta",
                    "player_id": 21,
                    "player_name": "FK Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Free Kick",
                    "period": 1,
                    "minute": 22,
                    "second": 0,
                    "location_x": 70,
                    "location_y": 58,
                    "pass_end_location_x": 75,
                    "pass_end_location_y": 58,
                    "pass_cross": False,
                    "pass_outcome_name": "Incomplete",
                },
            ]
        )

        out = extract_set_piece_events(
            events,
            include_follow_up=True,
            follow_up_seconds=15,
            next_n_actions=5,
            counting_mode="phase_events",
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(list(out.columns), SET_PIECE_OUTPUT_COLUMNS)

        first = out.iloc[0]
        self.assertEqual(first["set_piece_type"], "Corner")
        self.assertEqual(first["team"], "Alpha")
        self.assertEqual(first["taker"], "Corner Taker")
        self.assertTrue(bool(first["linked_shot"]))
        self.assertTrue(bool(first["linked_goal"]))

        third = out.iloc[2]
        self.assertEqual(third["set_piece_type"], "Free Kick")
        self.assertTrue(bool(third["short_set_piece"]))

        checks = compute_set_piece_sanity_checks(out)
        self.assertEqual(checks["total_rows"], 3)
        self.assertEqual(checks["linked_shot_total"], 1)
        self.assertEqual(checks["linked_goal_total"], 1)
        self.assertEqual(int(checks["counts_by_set_piece_type"]["events"].sum()), 3)

    def test_restart_only_vs_phase_events_counting(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 100,
                    "match_id": 7,
                    "event_index": 10,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 11,
                    "player_name": "FK Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Free Kick",
                    "period": 1,
                    "minute": 5,
                    "second": 0,
                },
                {
                    "event_id": 101,
                    "match_id": 7,
                    "event_index": 11,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 12,
                    "player_name": "Receiver",
                    "type_name": "Pass",
                    "play_pattern_name": "From Free Kick",
                    "period": 1,
                    "minute": 5,
                    "second": 3,
                },
                {
                    "event_id": 102,
                    "match_id": 7,
                    "event_index": 12,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 12,
                    "player_name": "Shooter",
                    "type_name": "Shot",
                    "play_pattern_name": "From Free Kick",
                    "period": 1,
                    "minute": 5,
                    "second": 7,
                },
                {
                    "event_id": 200,
                    "match_id": 7,
                    "event_index": 40,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 13,
                    "player_name": "FK Taker 2",
                    "type_name": "Pass",
                    "play_pattern_name": "From Free Kick",
                    "period": 1,
                    "minute": 15,
                    "second": 0,
                },
            ]
        )

        restart_only = extract_set_piece_events(events, counting_mode="restart_only")
        phase_events = extract_set_piece_events(events, counting_mode="phase_events")

        self.assertEqual(len(restart_only), 3)
        self.assertEqual(len(phase_events), 4)
        self.assertTrue(set(restart_only["event_id"].tolist()).issubset(set(phase_events["event_id"].tolist())))
        self.assertEqual(restart_only["restart_event_reason"].tolist(), ["First in sequence", "Direct shot type", "Index gap"])

    def test_restart_reason_direct_shot_type_takes_priority(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 401,
                    "match_id": 8,
                    "event_index": 10,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 11,
                    "player_name": "FK Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Free Kick",
                    "period": 1,
                    "minute": 5,
                    "second": 0,
                },
                {
                    "event_id": 402,
                    "match_id": 8,
                    "event_index": 11,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 12,
                    "player_name": "Shooter",
                    "type_name": "Shot",
                    "play_pattern_name": "From Free Kick",
                    "period": 1,
                    "minute": 5,
                    "second": 4,
                },
            ]
        )

        phase_events = extract_set_piece_events(events, counting_mode="phase_events")
        reasons = dict(zip(phase_events["event_id"], phase_events["restart_event_reason"], strict=False))
        self.assertEqual(reasons[401], "First in sequence")
        self.assertEqual(reasons[402], "Direct shot type")

    def test_dedupe_uses_stable_event_key(self) -> None:
        base = {
            "event_id": 301,
            "match_id": 9,
            "event_index": 25,
            "team_id": 1,
            "team_name": "Alpha",
            "player_id": 99,
            "player_name": "Corner Taker",
            "type_name": "Pass",
            "play_pattern_name": "From Corner",
            "period": 1,
            "minute": 30,
            "second": 0,
        }
        with_event_id_dupes = pd.DataFrame([base, dict(base), dict(base)])
        out_with_id = extract_set_piece_events(with_event_id_dupes, counting_mode="phase_events")
        self.assertEqual(len(out_with_id), 1)
        self.assertEqual(out_with_id.iloc[0]["event_key"], "m:9|eid:301")

        without_id = dict(base)
        without_id["event_id"] = pd.NA
        fallback_dupes = pd.DataFrame([without_id, dict(without_id)])
        out_without_id = extract_set_piece_events(fallback_dupes, counting_mode="phase_events")
        self.assertEqual(len(out_without_id), 1)
        self.assertTrue(str(out_without_id.iloc[0]["event_key"]).startswith("m:9|idx:25"))

    def test_extract_defensive_corner_clearances_and_lane_summary(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 100,
                    "match_id": 55,
                    "event_index": 10,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 11,
                    "player_name": "Corner Taker",
                    "type_name": "Pass",
                    "play_pattern_name": "From Corner",
                    "period": 1,
                    "minute": 10,
                    "second": 0,
                    "location_x": 120,
                    "location_y": 80,
                    "pass_end_location_x": 112,
                    "pass_end_location_y": 36,
                },
                {
                    "event_id": 101,
                    "match_id": 55,
                    "event_index": 11,
                    "team_id": 2,
                    "team_name": "Beta",
                    "player_id": 21,
                    "player_name": "Clearer One",
                    "type_name": "Clearance",
                    "play_pattern_name": "From Corner",
                    "period": 1,
                    "minute": 10,
                    "second": 2,
                    "location_x": 8,
                    "location_y": 40,
                },
                {
                    "event_id": 102,
                    "match_id": 55,
                    "event_index": 12,
                    "team_id": 2,
                    "team_name": "Beta",
                    "player_id": 22,
                    "player_name": "Pressing Mid",
                    "type_name": "Pressure",
                    "play_pattern_name": "From Corner",
                    "period": 1,
                    "minute": 10,
                    "second": 3,
                    "location_x": 11,
                    "location_y": 38,
                },
                {
                    "event_id": 103,
                    "match_id": 55,
                    "event_index": 13,
                    "team_id": 1,
                    "team_name": "Alpha",
                    "player_id": 12,
                    "player_name": "Recovery Player",
                    "type_name": "Ball Recovery",
                    "play_pattern_name": "From Corner",
                    "period": 1,
                    "minute": 10,
                    "second": 5,
                    "location_x": 96,
                    "location_y": 16,
                },
                {
                    "event_id": 104,
                    "match_id": 55,
                    "event_index": 20,
                    "team_id": 2,
                    "team_name": "Beta",
                    "player_id": 23,
                    "player_name": "Clearer Two",
                    "type_name": "Clearance",
                    "play_pattern_name": "From Corner",
                    "period": 2,
                    "minute": 48,
                    "second": 0,
                    "location_x": 10,
                    "location_y": 42,
                },
                {
                    "event_id": 105,
                    "match_id": 55,
                    "event_index": 21,
                    "team_id": 2,
                    "team_name": "Beta",
                    "player_id": 24,
                    "player_name": "Outlet Passer",
                    "type_name": "Pass",
                    "play_pattern_name": "From Corner",
                    "period": 2,
                    "minute": 48,
                    "second": 4,
                    "location_x": 66,
                    "location_y": 64,
                    "pass_end_location_x": 82,
                    "pass_end_location_y": 60,
                },
                {
                    "event_id": 106,
                    "match_id": 55,
                    "event_index": 30,
                    "team_id": 2,
                    "team_name": "Beta",
                    "player_id": 25,
                    "player_name": "Clearer Three",
                    "type_name": "Clearance",
                    "play_pattern_name": "From Corner",
                    "period": 2,
                    "minute": 60,
                    "second": 0,
                    "location_x": 9,
                    "location_y": 39,
                },
                {
                    "event_id": 107,
                    "match_id": 55,
                    "event_index": 31,
                    "team_id": 2,
                    "team_name": "Beta",
                    "player_id": 26,
                    "player_name": "Bench Player",
                    "type_name": "Substitution",
                    "play_pattern_name": "From Corner",
                    "period": 2,
                    "minute": 60,
                    "second": 3,
                },
            ]
        )

        out = extract_defensive_corner_clearances(events, follow_up_seconds=12, next_n_actions=4)
        self.assertEqual(len(out), 3)

        first = out.iloc[0]
        self.assertEqual(first["exit_lane"], "Left")
        self.assertEqual(first["first_ball_winner"], "Attacking team")
        self.assertEqual(first["exit_event_type"], "Ball Recovery")

        second = out.iloc[1]
        self.assertEqual(second["exit_lane"], "Right")
        self.assertEqual(second["first_ball_winner"], "Defending team")

        third = out.iloc[2]
        self.assertEqual(third["exit_lane"], "Unknown")
        self.assertEqual(third["first_ball_winner"], "Unknown")

        summary = summarize_defensive_corner_clearances(out)
        self.assertEqual(summary["clearances"].sum(), 3)
        left_row = summary[summary["exit_lane"] == "Left"].iloc[0]
        right_row = summary[summary["exit_lane"] == "Right"].iloc[0]
        self.assertAlmostEqual(float(left_row["share_pct"]), 33.3, places=1)
        self.assertEqual(int(right_row["defending_team_first_ball"]), 1)
        self.assertEqual(float(right_row["defending_team_first_ball_pct"]), 100.0)


if __name__ == "__main__":
    unittest.main()
