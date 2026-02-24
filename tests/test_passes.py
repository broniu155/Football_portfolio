import unittest

import pandas as pd

from app.components.passes import compute_pass_stats, filter_passes, resolve_pass_columns


class PassesTests(unittest.TestCase):
    def test_resolve_pass_columns_aliases(self) -> None:
        df = pd.DataFrame(
            {
                "pass.height.name": ["Ground Pass"],
                "pass.cross": [True],
                "pass.outcome.name": [None],
            }
        )
        colmap = resolve_pass_columns(df)
        self.assertEqual(colmap["height_name_col"], "pass.height.name")
        self.assertEqual(colmap["cross_col"], "pass.cross")
        self.assertEqual(colmap["outcome_name_col"], "pass.outcome.name")

    def test_filter_passes_combined_filters(self) -> None:
        df = pd.DataFrame(
            {
                "pass_height_norm": ["ground pass", "high pass", "low pass"],
                "is_cross": [False, True, True],
                "is_completed": [True, False, True],
                "location_x": [40, 50, 30],
                "pass_end_location_x": [70, 55, 60],
            }
        )
        out = filter_passes(df, pass_height_filter="High", cross_filter="Crosses Only", final_third_only=False, is_home=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["pass_height_norm"], "high pass")

    def test_compute_pass_stats_na_when_missing_breakdown_fields(self) -> None:
        df = pd.DataFrame(
            {
                "is_completed": [True, False],
                "location_x": [40, 50],
                "pass_end_location_x": [70, 60],
            }
        )
        stats = compute_pass_stats(df, is_home=True)
        self.assertIsNone(stats["ground_passes"])
        self.assertIsNone(stats["crosses"])
        self.assertEqual(stats["total_passes"], 2)


if __name__ == "__main__":
    unittest.main()
