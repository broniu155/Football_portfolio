import unittest

import pandas as pd

from app.components.lineups import get_formation, get_starting_positions, get_starting_xi, get_unmapped_position_names


class LineupsSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = [
            {"match_id": 1, "team_id": 10, "team_name": "Home", "type_name": "Starting XI", "event_index": 1, "minute": 0, "second": 0},
            {"match_id": 1, "team_id": 20, "team_name": "Away", "type_name": "Starting XI", "event_index": 2, "minute": 0, "second": 0},
        ]
        for i in range(1, 16):
            if i == 2:
                pos = "Right Center Back"
            elif i == 3:
                pos = "Center Forward"
            elif i == 4:
                pos = "Mystery Role"
            else:
                pos = "Center Midfield" if i > 1 else "Goalkeeper"
            rows.append(
                {
                    "match_id": 1,
                    "team_id": 10,
                    "team_name": "Home",
                    "type_name": "Pass",
                    "event_index": 10 + i,
                    "minute": i,
                    "second": 0,
                    "player_id": i,
                    "player_name": f"Home Player {i}",
                    "position_name": pos,
                }
            )
        self.events = pd.DataFrame(rows)

    def test_starting_xi_and_positions_smoke(self) -> None:
        xi = get_starting_xi(self.events, match_id=1, team_id=10)
        self.assertGreater(len(xi), 0)
        self.assertLessEqual(len(xi), 11)
        self.assertFalse(xi["player_name"].astype(str).str.strip().eq("").all())

        formation = get_formation(self.events, match_id=1, team_id=10)
        positions = get_starting_positions(self.events, match_id=1, team_id=10, formation=formation)
        self.assertGreater(len(positions), 0)
        self.assertLessEqual(len(positions), 11)

    def test_position_mapping_and_unmapped_positions(self) -> None:
        positions = get_starting_positions(self.events, match_id=1, team_id=10, is_home=True)
        by_name = {row["position_name"]: row for row in positions}
        self.assertIn("Center Forward", by_name)
        self.assertIn("Right Center Back", by_name)
        # Home CF should be further upfield than home RCB.
        self.assertGreater(float(by_name["Center Forward"]["y"]), float(by_name["Right Center Back"]["y"]))

        # Home and away keepers must be on opposite ends.
        home = get_starting_positions(self.events, match_id=1, team_id=10, is_home=True)
        away = get_starting_positions(self.events, match_id=1, team_id=10, is_home=False)
        home_gk = next((row for row in home if row["position_name"] == "Goalkeeper"), None)
        away_gk = next((row for row in away if row["position_name"] == "Goalkeeper"), None)
        self.assertIsNotNone(home_gk)
        self.assertIsNotNone(away_gk)
        self.assertLess(float(home_gk["y"]), 50.0)
        self.assertGreater(float(away_gk["y"]), 50.0)

        unmapped = get_unmapped_position_names(match_id=1, fact_events=self.events, team_id=10)
        self.assertIn("Mystery Role", unmapped)


if __name__ == "__main__":
    unittest.main()
