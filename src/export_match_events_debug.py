from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.export import build_match_events_export_df


def _read_table(base_dir: Path, table: str) -> pd.DataFrame:
    parquet = base_dir / f"{table}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv = base_dir / f"{table}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {table}.parquet or {table}.csv in {base_dir}")


def _source_coord_summary(df: pd.DataFrame) -> pd.DataFrame:
    source_defs = {
        "event.location": ("location_x", "location_y", None),
        "pass.end_location": ("pass_end_location_x", "pass_end_location_y", "pass"),
        "carry.end_location": ("carry_end_location_x", "carry_end_location_y", "carry"),
        "shot.end_location": ("shot_end_location_x", "shot_end_location_y", "shot"),
    }
    event_type = df.get("type_name", pd.Series("", index=df.index)).astype("string").str.strip().str.lower()
    rows: list[dict[str, object]] = []
    for source, (x_col, y_col, only_type) in source_defs.items():
        if x_col in df.columns:
            x = pd.to_numeric(df[x_col], errors="coerce")
        else:
            x = pd.Series(pd.NA, index=df.index, dtype="float64")
        if y_col in df.columns:
            y = pd.to_numeric(df[y_col], errors="coerce")
        else:
            y = pd.Series(pd.NA, index=df.index, dtype="float64")
        mask = pd.Series(True, index=df.index)
        if only_type is not None:
            mask = event_type.eq(only_type)
        x_m = x.loc[mask]
        y_m = y.loc[mask]
        missing = x_m.isna() | y_m.isna()
        out_of_range = (~missing) & (~(x_m.between(0.0, 120.0, inclusive="both") & y_m.between(0.0, 80.0, inclusive="both")))
        rows.append(
            {
                "source": source,
                "scope_rows": int(mask.sum()),
                "missing_coords": int(missing.sum()),
                "out_of_range": int(out_of_range.sum()),
            }
        )
    return pd.DataFrame(rows)


def _print_counts(title: str, series: pd.Series) -> None:
    print(f"\n{title}")
    for key, value in series.items():
        print(f"- {key}: {int(value)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export match events CSV with attack-channel debug columns.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_model_sample"))
    parser.add_argument("--match-id", type=int, required=True)
    parser.add_argument("--team-id", type=int, default=None)
    parser.add_argument("--player-id", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs") / "match_events_debug.csv")
    parser.add_argument("--essential-only", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = _read_table(args.data_dir, "fact_events")
    if "match_id" not in events.columns:
        raise ValueError("fact_events is missing match_id")

    scoped = events[pd.to_numeric(events["match_id"], errors="coerce") == int(args.match_id)].copy()
    if args.team_id is not None and "team_id" in scoped.columns:
        scoped = scoped[pd.to_numeric(scoped["team_id"], errors="coerce") == int(args.team_id)]
    if args.player_id is not None and "player_id" in scoped.columns:
        scoped = scoped[pd.to_numeric(scoped["player_id"], errors="coerce") == int(args.player_id)]

    export_df = build_match_events_export_df(scoped, include_derived=True, essential_only=bool(args.essential_only))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(export_df):,} rows to {args.output_csv}")
    print(f"Total rows: {len(export_df):,}")
    dribble_mask = export_df.get("dribble_is_attempt", pd.Series(False, index=export_df.index)).fillna(False).astype(bool)
    print(f"Dribble rows: {int(dribble_mask.sum()):,}")
    if dribble_mask.any():
        dribble_outcomes = (
            export_df.get("dribble_outcome_name", pd.Series(pd.NA, index=export_df.index))
            .astype("string")
            .fillna("Unknown")
            .replace("", "Unknown")
        )
        _print_counts("Dribble outcomes", dribble_outcomes.loc[dribble_mask].value_counts(dropna=False))
        top_players = (
            export_df.get("player_name", pd.Series("Unknown Player", index=export_df.index))
            .astype("string")
            .fillna("Unknown Player")
            .loc[dribble_mask]
            .value_counts(dropna=False)
            .head(10)
        )
        _print_counts("Top 10 players by dribble attempts", top_players)
    if {"analysis_group", "analysis_subgroup"}.issubset(export_df.columns):
        _print_counts(
            "Counts by analysis_group",
            export_df["analysis_group"].astype("string").fillna("other").value_counts(dropna=False),
        )
        _print_counts(
            "Counts by analysis_subgroup",
            export_df["analysis_subgroup"].astype("string").fillna("other").value_counts(dropna=False),
        )
        group_total = int(
            export_df["analysis_group"].astype("string").fillna("other").value_counts(dropna=False).sum()
        )
        print(f"Sanity check: total_rows={len(export_df):,}, sum(groups)={group_total:,}")
    if "attack_channel" in export_df.columns:
        _print_counts("Counts by attack_channel", export_df["attack_channel"].astype("string").fillna("Unknown").value_counts(dropna=False))
    if "channel_source" in export_df.columns:
        _print_counts("Counts by channel_source", export_df["channel_source"].astype("string").fillna("none").value_counts(dropna=False))
    if "channel_reason" in export_df.columns:
        _print_counts("Counts by channel_reason", export_df["channel_reason"].astype("string").fillna("missing").value_counts(dropna=False))

    summary = _source_coord_summary(scoped)
    print("\nCoordinate diagnostics by source")
    for _, row in summary.iterrows():
        print(
            f"- {row['source']}: scope_rows={int(row['scope_rows'])}, "
            f"missing_coords={int(row['missing_coords'])}, out_of_range={int(row['out_of_range'])}"
        )


if __name__ == "__main__":
    main()
