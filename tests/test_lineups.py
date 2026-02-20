import unittest

import pandas as pd

from app.components.lineups import get_formation, get_starting_positions, get_starting_xi


class LineupsSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = [
            {"match_id": 1, "team_id": 10, "team_name": "Home", "type_name": "Starting XI", "event_index": 1, "minute": 0, "second": 0},
            {"match_id": 1, "team_id": 20, "team_name": "Away", "type_name": "Starting XI", "event_index": 2, "minute": 0, "second": 0},
        ]
        for i in range(1, 16):
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
                    "position_name": "Midfielder" if i > 1 else "Goalkeeper",
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


if __name__ == "__main__":
    unittest.main()
