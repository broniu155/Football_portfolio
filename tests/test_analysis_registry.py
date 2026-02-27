import unittest

import pandas as pd

from app.components.analysis_registry import ANALYSIS_VIEWS, classify_analysis_groups


class AnalysisRegistryTests(unittest.TestCase):
    def test_analysis_view_order(self) -> None:
        self.assertEqual(
            ANALYSIS_VIEWS,
            ("Stats", "Offensive", "Passes", "Transitions", "Defensive", "Set Pieces", "More"),
        )

    def test_event_classification_groups_and_subgroups(self) -> None:
        df = pd.DataFrame(
            [
                {"type_name": "Shot"},
                {"type_name": "Pass"},
                {"type_name": "Carry"},
                {"type_name": "Interception"},
                {"type_name": "Duel", "duel_type_name": "Offensive Duel", "bucket": "OFFENSIVE"},
                {"type_name": "Duel", "duel_type_name": "Tackle", "bucket": "DEFENSIVE"},
            ]
        )
        out = classify_analysis_groups(df)
        self.assertEqual(out["analysis_group"].tolist(), ["offensive", "passes", "transitions", "defensive", "offensive", "defensive"])
        self.assertEqual(out["analysis_subgroup"].tolist(), ["shots", "passes", "carries", "recoveries", "duels_offensive", "duels_defensive"])


if __name__ == "__main__":
    unittest.main()
