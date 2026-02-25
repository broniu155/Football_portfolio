import unittest

import pandas as pd

from app.components.event_classification import EventBucket, derive_event_labels


class EventClassificationTests(unittest.TestCase):
    def test_bucket_priority_set_piece_over_transition(self) -> None:
        events = pd.DataFrame(
            [
                {"type_name": "Miscontrol", "play_pattern_name": "From Corner", "counterpress": True},
                {"type_name": "Pass", "play_pattern_name": "Regular Play", "counterpress": True},
            ]
        )
        out = derive_event_labels(events)
        self.assertEqual(out.iloc[0]["bucket"], EventBucket.SET_PIECE.value)
        self.assertEqual(out.iloc[1]["bucket"], EventBucket.TRANSITION.value)

    def test_missing_columns_is_safe(self) -> None:
        events = pd.DataFrame([{"match_id": 1}, {"match_id": 1}])
        out = derive_event_labels(events)
        self.assertEqual(len(out), 2)
        self.assertTrue({"bucket", "subtype", "is_turnover", "is_regain", "is_set_piece"}.issubset(out.columns))

    def test_ball_receipt_incomplete_is_turnover(self) -> None:
        events = pd.DataFrame([{"type_name": "Ball Receipt", "ball_receipt_outcome_name": "Incomplete"}])
        out = derive_event_labels(events)
        self.assertTrue(bool(out.iloc[0]["is_turnover"]))
        self.assertEqual(out.iloc[0]["bucket"], EventBucket.TRANSITION.value)


if __name__ == "__main__":
    unittest.main()
