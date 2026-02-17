from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_sample_tables() -> dict[str, pd.DataFrame]:
    dim_match = pd.DataFrame(
        [
            {
                "match_id": 1001,
                "competition_id": 11,
                "season_id": 2023,
                "match_date": "2023-10-01",
                "kick_off": "15:00:00",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_name": "North City",
                "away_team_name": "South United",
                "home_score": 2,
                "away_score": 1,
                "stadium_name": "North Park",
                "referee_name": "A. Ref",
                "match_week": 7,
                "match_status": "available",
                "last_updated": "2023-10-02",
                "last_updated_360": "",
                "match_status_360": "",
                "match_label": "North City vs South United (2023-10-01)",
                "is_360_available": 0,
                "competition_name": "Sample League",
                "season_name": "2023/24",
            },
            {
                "match_id": 1002,
                "competition_id": 11,
                "season_id": 2023,
                "match_date": "2023-10-08",
                "kick_off": "17:30:00",
                "home_team_id": 3,
                "away_team_id": 4,
                "home_team_name": "East FC",
                "away_team_name": "West Rovers",
                "home_score": 0,
                "away_score": 0,
                "stadium_name": "East Arena",
                "referee_name": "B. Ref",
                "match_week": 8,
                "match_status": "available",
                "last_updated": "2023-10-09",
                "last_updated_360": "",
                "match_status_360": "",
                "match_label": "East FC vs West Rovers (2023-10-08)",
                "is_360_available": 0,
                "competition_name": "Sample League",
                "season_name": "2023/24",
            },
        ]
    )

    dim_team = pd.DataFrame(
        [
            {"team_id": 1, "team_name": "North City"},
            {"team_id": 2, "team_name": "South United"},
            {"team_id": 3, "team_name": "East FC"},
            {"team_id": 4, "team_name": "West Rovers"},
        ]
    )

    dim_player = pd.DataFrame(
        [
            {"player_id": 11, "player_name": "Noah Striker", "team_id": 1},
            {"player_id": 12, "player_name": "Liam Mid", "team_id": 1},
            {"player_id": 21, "player_name": "Ethan Forward", "team_id": 2},
            {"player_id": 22, "player_name": "Mason Wing", "team_id": 2},
            {"player_id": 31, "player_name": "Lucas Nine", "team_id": 3},
            {"player_id": 41, "player_name": "Owen Keeper", "team_id": 4},
        ]
    )

    fact_events_rows = []
    event_id = 1
    for match_id, home_team_id, away_team_id, home_team_name, away_team_name in [
        (1001, 1, 2, "North City", "South United"),
        (1002, 3, 4, "East FC", "West Rovers"),
    ]:
        for minute in range(1, 21):
            fact_events_rows.append(
                {
                    "event_id": event_id,
                    "match_id": match_id,
                    "team_id": home_team_id if minute % 2 == 0 else away_team_id,
                    "team_name": home_team_name if minute % 2 == 0 else away_team_name,
                    "player_id": 11 if match_id == 1001 else 31,
                    "player_name": "Noah Striker" if match_id == 1001 else "Lucas Nine",
                    "type_id": 30 if minute % 5 else 16,
                    "type_name": "Pass" if minute % 5 else "Shot",
                    "period": 1 if minute <= 10 else 2,
                    "minute": minute,
                    "second": 0,
                    "location_x": 20 + minute * 2,
                    "location_y": 15 + (minute % 10) * 3,
                    "under_pressure": 1 if minute % 4 == 0 else 0,
                    "counterpress": 1 if minute % 7 == 0 else 0,
                }
            )
            event_id += 1

    fact_events = pd.DataFrame(fact_events_rows)

    fact_shots = pd.DataFrame(
        [
            {
                "event_id": 5,
                "match_id": 1001,
                "team_id": 1,
                "team_name": "North City",
                "player_id": 11,
                "player_name": "Noah Striker",
                "minute": 5,
                "second": 12,
                "period": 1,
                "x": 98,
                "y": 42,
                "location_x": 98,
                "location_y": 42,
                "shot_end_location_x": 120,
                "shot_end_location_y": 40,
                "xg": 0.18,
                "shot_statsbomb_xg": 0.18,
                "shot_outcome": "Saved",
                "shot_outcome_id": 96,
                "under_pressure": 1,
            },
            {
                "event_id": 10,
                "match_id": 1001,
                "team_id": 2,
                "team_name": "South United",
                "player_id": 21,
                "player_name": "Ethan Forward",
                "minute": 14,
                "second": 3,
                "period": 1,
                "x": 105,
                "y": 30,
                "location_x": 105,
                "location_y": 30,
                "shot_end_location_x": 120,
                "shot_end_location_y": 34,
                "xg": 0.32,
                "shot_statsbomb_xg": 0.32,
                "shot_outcome": "Goal",
                "shot_outcome_id": 97,
                "under_pressure": 0,
            },
            {
                "event_id": 25,
                "match_id": 1002,
                "team_id": 3,
                "team_name": "East FC",
                "player_id": 31,
                "player_name": "Lucas Nine",
                "minute": 12,
                "second": 48,
                "period": 2,
                "x": 92,
                "y": 55,
                "location_x": 92,
                "location_y": 55,
                "shot_end_location_x": 120,
                "shot_end_location_y": 60,
                "xg": 0.09,
                "shot_statsbomb_xg": 0.09,
                "shot_outcome": "Off T",
                "shot_outcome_id": 98,
                "under_pressure": 1,
            },
        ]
    )

    return {
        "dim_match": dim_match,
        "dim_team": dim_team,
        "dim_player": dim_player,
        "fact_events": fact_events,
        "fact_shots": fact_shots,
    }


def write_tables(output_dir: Path, file_format: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_sample_tables()
    for name, df in tables.items():
        if file_format in {"csv", "both"}:
            df.to_csv(output_dir / f"{name}.csv", index=False)
        if file_format in {"parquet", "both"}:
            df.to_parquet(output_dir / f"{name}.parquet", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tiny sample star-schema files for Streamlit.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "app" / "data" / "sample_star_schema",
    )
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_tables(args.output_dir, args.format)
    print(f"Wrote sample star schema to {args.output_dir} ({args.format})")


if __name__ == "__main__":
    main()
