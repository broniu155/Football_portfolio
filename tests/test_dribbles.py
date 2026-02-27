import unittest
from pathlib import Path

import pandas as pd

from app.components.dribbles import prepare_dribble_events, summarize_dribbles


class DribbleParsingTests(unittest.TestCase):
    def test_incomplete_outcome_counts_as_incomplete(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "event_id": "a1",
                    "match_id": 1,
                    "type_name": "Dribble",
                    "dribble_outcome_name": "Incomplete",
                }
            ]
        )
        out, _ = prepare_dribble_events(df)
        self.assertEqual(len(out), 1)
        self.assertFalse(bool(out.iloc[0]["dribble_is_complete"]))
        self.assertTrue(bool(out.iloc[0]["dribble_is_incomplete"]))
        self.assertEqual(out.iloc[0]["dribble_outcome_resolved"], "Incomplete")

    def test_complete_outcome_counts_as_complete(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "event_id": "a2",
                    "match_id": 1,
                    "type_name": "Dribble",
                    "dribble_outcome_name": "Complete",
                }
            ]
        )
        out, _ = prepare_dribble_events(df)
        self.assertEqual(len(out), 1)
        self.assertTrue(bool(out.iloc[0]["dribble_is_complete"]))
        self.assertFalse(bool(out.iloc[0]["dribble_is_incomplete"]))
        self.assertEqual(out.iloc[0]["dribble_outcome_resolved"], "Complete")

    def test_missing_outcome_is_unknown_not_complete(self) -> None:
        df = pd.DataFrame([{"event_id": "a3", "match_id": 1, "type_name": "Dribble"}])
        out, _ = prepare_dribble_events(df)
        self.assertEqual(len(out), 1)
        self.assertFalse(bool(out.iloc[0]["dribble_is_complete"]))
        self.assertFalse(bool(out.iloc[0]["dribble_is_incomplete"]))
        self.assertEqual(out.iloc[0]["dribble_outcome_resolved"], "Unknown")

    def test_sample_match_regression_count(self) -> None:
        sample_path = Path("data_model_sample/fact_events.parquet")
        if not sample_path.exists():
            self.skipTest("sample parquet fixture unavailable")
        sample = pd.read_parquet(sample_path, columns=["match_id", "event_id", "type_name"])
        match_id = 3895266
        match_rows = sample[pd.to_numeric(sample["match_id"], errors="coerce") == match_id].copy()
        if match_rows.empty:
            self.skipTest(f"match {match_id} not available in sample fixture")
        summary = summarize_dribbles(match_rows)
        self.assertEqual(summary["total_dribble_events"], 20)
        self.assertEqual(summary["duplicates_removed"], 0)


if __name__ == "__main__":
    unittest.main()
