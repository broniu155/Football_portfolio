from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.export_star_schema import build_star_schema

REQUIRED_TABLES = ("dim_match", "dim_team", "dim_player", "fact_events", "fact_shots")
OPTIONAL_TABLES = (
    "dim_competition",
    "dim_season",
    "dim_shot_outcome",
    "dim_body_part",
    "dim_shot_type",
)


def _read_table(base_dir: Path, table: str) -> pd.DataFrame:
    parquet = base_dir / f"{table}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv = base_dir / f"{table}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing table: {table}.parquet or {table}.csv")


def _comp_and_season_ids(
    model_dir: Path,
    competition_name: str,
    season_name: str,
) -> tuple[int, int]:
    dim_comp = _read_table(model_dir, "dim_competition")
    dim_season = _read_table(model_dir, "dim_season")

    comp_name_col = next((c for c in dim_comp.columns if "competition_name" in c.lower()), None)
    season_name_col = next((c for c in dim_season.columns if "season_name" in c.lower()), None)
    if comp_name_col is None or season_name_col is None:
        raise ValueError("Could not find competition_name/season_name columns in dimension tables.")

    comp_series = dim_comp[comp_name_col].astype("string").str.strip().str.lower()
    season_series = dim_season[season_name_col].astype("string").str.strip().str.lower()
    comp_target = competition_name.strip().lower()
    season_target = season_name.strip().lower()

    comp_rows = dim_comp[comp_series == comp_target]
    if comp_rows.empty:
        comp_rows = dim_comp[comp_series.str.contains(comp_target, na=False)]
    season_rows = dim_season[season_series == season_target]
    if comp_rows.empty:
        raise ValueError(f"Competition not found: {competition_name}")
    if season_rows.empty:
        raise ValueError(f"Season not found: {season_name}")

    comp_id = int(pd.to_numeric(comp_rows.iloc[0]["competition_id"], errors="coerce"))
    season_id = int(pd.to_numeric(season_rows.iloc[0]["season_id"], errors="coerce"))
    return comp_id, season_id


def _select_matches(dim_match: pd.DataFrame, competition_id: int, season_id: int, n_matches: int) -> pd.DataFrame:
    work = dim_match.copy()
    work["competition_id"] = pd.to_numeric(work["competition_id"], errors="coerce")
    work["season_id"] = pd.to_numeric(work["season_id"], errors="coerce")
    scoped = work[(work["competition_id"] == competition_id) & (work["season_id"] == season_id)].copy()
    if scoped.empty:
        raise ValueError("No matches found for requested competition/season.")

    sort_cols = [c for c in ("match_date", "match_id") if c in scoped.columns]
    if sort_cols:
        scoped = scoped.sort_values(sort_cols, ascending=False)
    return scoped.head(max(1, int(n_matches))).copy()


def _ensure_model_exists(model_dir: Path, processed_dir: Path) -> None:
    needed = [model_dir / f"{t}.parquet" for t in REQUIRED_TABLES]
    if all(path.exists() for path in needed):
        return
    if not processed_dir.exists():
        raise FileNotFoundError(f"Model not found in {model_dir} and processed input is missing: {processed_dir}")
    build_star_schema(
        input_dir=processed_dir,
        output_dir=model_dir,
        output_format="parquet",
    )


def build_sample(
    model_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    competition_name: str,
    season_name: str,
    n_matches: int,
) -> None:
    _ensure_model_exists(model_dir=model_dir, processed_dir=processed_dir)

    dim_match = _read_table(model_dir, "dim_match")
    fact_events = _read_table(model_dir, "fact_events")
    fact_shots = _read_table(model_dir, "fact_shots")
    dim_team = _read_table(model_dir, "dim_team")
    dim_player = _read_table(model_dir, "dim_player")

    competition_id, season_id = _comp_and_season_ids(
        model_dir=model_dir,
        competition_name=competition_name,
        season_name=season_name,
    )
    dim_match_sample = _select_matches(
        dim_match=dim_match,
        competition_id=competition_id,
        season_id=season_id,
        n_matches=n_matches,
    )

    match_ids = (
        pd.to_numeric(dim_match_sample["match_id"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    match_id_set = set(match_ids)

    fact_events["match_id"] = pd.to_numeric(fact_events["match_id"], errors="coerce")
    fact_shots["match_id"] = pd.to_numeric(fact_shots["match_id"], errors="coerce")
    fact_events_sample = fact_events[fact_events["match_id"].isin(match_id_set)].copy()
    fact_shots_sample = fact_shots[fact_shots["match_id"].isin(match_id_set)].copy()

    team_ids = set()
    for source_df, cols in (
        (dim_match_sample, ("home_team_id", "away_team_id")),
        (fact_events_sample, ("team_id",)),
        (fact_shots_sample, ("team_id",)),
    ):
        for col in cols:
            if col in source_df.columns:
                values = pd.to_numeric(source_df[col], errors="coerce").dropna().astype(int).tolist()
                team_ids.update(values)

    player_ids = set()
    for source_df in (fact_events_sample, fact_shots_sample):
        if "player_id" in source_df.columns:
            values = pd.to_numeric(source_df["player_id"], errors="coerce").dropna().astype(int).tolist()
            player_ids.update(values)

    dim_team["team_id"] = pd.to_numeric(dim_team["team_id"], errors="coerce")
    dim_player["player_id"] = pd.to_numeric(dim_player["player_id"], errors="coerce")
    dim_team_sample = dim_team[dim_team["team_id"].isin(team_ids)].copy()
    dim_player_sample = dim_player[dim_player["player_id"].isin(player_ids)].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_to_write: dict[str, pd.DataFrame] = {
        "dim_match": dim_match_sample,
        "dim_team": dim_team_sample,
        "dim_player": dim_player_sample,
        "fact_events": fact_events_sample,
        "fact_shots": fact_shots_sample,
    }

    for optional in OPTIONAL_TABLES:
        try:
            tables_to_write[optional] = _read_table(model_dir, optional)
        except FileNotFoundError:
            continue

    for table_name, table_df in tables_to_write.items():
        table_df.to_parquet(output_dir / f"{table_name}.parquet", index=False)

    manifest = {
        "competition": competition_name,
        "season": season_name,
        "n_matches": len(match_ids),
        "match_ids": match_ids,
        "row_counts": {name: int(len(df)) for name, df in tables_to_write.items()},
    }
    (output_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated sample star-schema dataset.")
    parser.add_argument("--model-dir", type=Path, default=Path("data_model"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data_processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_model_sample"))
    parser.add_argument("--competition", default="Bundesliga")
    parser.add_argument("--season", default="2023/2024")
    parser.add_argument("--n-matches", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_sample(
        model_dir=args.model_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        competition_name=args.competition,
        season_name=args.season,
        n_matches=args.n_matches,
    )
    print(f"Wrote sample model to {args.output_dir}")


if __name__ == "__main__":
    main()
