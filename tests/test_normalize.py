import unittest

from src.normalize import normalize_events, validate_coordinate_bounds


class NormalizeTests(unittest.TestCase):
    def test_home_and_away_attack_in_same_direction(self) -> None:
        rows = [
            {
                "team_id": "100",
                "location_x": "30",
                "location_y": "40",
                "pass_end_location_x": "90",
                "pass_end_location_y": "50",
            },
            {
                "team_id": "200",
                "location_x": "90",
                "location_y": "40",
                "pass_end_location_x": "30",
                "pass_end_location_y": "50",
            },
        ]

        normalized = normalize_events(rows, home_team_id=100)

        # Home team: 30/120 => 25.0; Away team: flip(90/120 => 75.0) => 25.0
        self.assertAlmostEqual(float(normalized[0]["location_x"]), 25.0)
        self.assertAlmostEqual(float(normalized[1]["location_x"]), 25.0)

        # End locations also align in same attacking direction.
        self.assertAlmostEqual(float(normalized[0]["pass_end_location_x"]), 75.0)
        self.assertAlmostEqual(float(normalized[1]["pass_end_location_x"]), 75.0)

    def test_no_negative_or_over_100_after_normalization(self) -> None:
        rows = [
            {
                "team_id": "100",
                "location_x": "-5",    # clamped to 0
                "location_y": "85",    # >80 -> clamped to 100 after scaling
                "pass_end_location_x": "130",  # >120 -> clamped to 100
                "pass_end_location_y": "-3",   # clamped to 0
            }
        ]

        normalized = normalize_events(rows, home_team_id=100)
        issues = validate_coordinate_bounds(normalized)

        self.assertEqual(issues, [])
        self.assertEqual(float(normalized[0]["location_x"]), 0.0)
        self.assertEqual(float(normalized[0]["pass_end_location_x"]), 100.0)


if __name__ == "__main__":
    unittest.main()
