from __future__ import annotations

import csv
from pathlib import Path


FACT_CANDIDATE_COLUMNS = [
    "event_id",
    "match_id",
    "event_index",
    "period",
    "minute",
    "second",
    "timestamp",
    "team_id",
    "player_id",
    "type_id",
    "type_name",
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
    "carry_end_location_x",
    "carry_end_location_y",
    "shot_end_location_x",
    "shot_end_location_y",
    "shot_statsbomb_xg",
    "shot_outcome_name",
    "pass_outcome_name",
    "under_pressure",
    "counterpress",
]

DIM_MATCH_COLUMNS = [
    "match_id",
    "team_1_id",
    "team_1_name",
    "team_2_id",
    "team_2_name",
    "event_count",
    "max_minute",
    "max_second",
]


def as_int(value: str) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def as_float(value: str) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_nested_like(value: str) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text.startswith("[") or text.startswith("{")


def build_star_schema(input_dir: Path, output_dir: Path) -> None:
    events_path = input_dir / "events.csv"
    teams_path = input_dir / "teams.csv"
    players_path = input_dir / "players.csv"

    if not events_path.exists():
        raise FileNotFoundError(f"Missing source file: {events_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fact_out = output_dir / "fact_events.csv"
    dim_match_out = output_dir / "dim_match.csv"
    dim_team_out = output_dir / "dim_team.csv"
    dim_player_out = output_dir / "dim_player.csv"

    match_meta: dict[int, dict[str, object]] = {}
    team_lookup: dict[int, str] = {}
    player_lookup: dict[int, tuple[str, int | None]] = {}

    with events_path.open("r", encoding="utf-8", newline="") as f_in, fact_out.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("events.csv has no header")

        source_cols = reader.fieldnames
        fact_cols = [c for c in FACT_CANDIDATE_COLUMNS if c in source_cols]

        required_fact = {"event_id", "match_id", "team_id", "player_id"}
        missing_req = sorted(required_fact - set(fact_cols))
        if missing_req:
            raise ValueError(f"Missing required fact columns: {missing_req}")

        writer = csv.DictWriter(f_out, fieldnames=fact_cols)
        writer.writeheader()

        for row in reader:
            flat_row: dict[str, str] = {}
            for col in fact_cols:
                val = row.get(col, "")
                flat_row[col] = "" if is_nested_like(val) else val
            writer.writerow(flat_row)

            match_id = as_int(row.get("match_id", ""))
            team_id = as_int(row.get("team_id", ""))
            player_id = as_int(row.get("player_id", ""))
            team_name = (row.get("team_name") or "").strip()
            player_name = (row.get("player_name") or "").strip()
            minute = as_int(row.get("minute", ""))
            second = as_int(row.get("second", ""))

            if match_id is None:
                continue

            if match_id not in match_meta:
                match_meta[match_id] = {
                    "teams": {},
                    "event_count": 0,
                    "max_minute": 0,
                    "max_second": 0,
                }

            mm = match_meta[match_id]
            mm["event_count"] = int(mm["event_count"]) + 1
            if minute is not None and minute > int(mm["max_minute"]):
                mm["max_minute"] = minute
            if second is not None and second > int(mm["max_second"]):
                mm["max_second"] = second

            teams = mm["teams"]
            if team_id is not None:
                teams[team_id] = team_name
                team_lookup[team_id] = team_name

            if player_id is not None:
                player_lookup[player_id] = (player_name, team_id)

    # Build dim_match
    with dim_match_out.open("w", encoding="utf-8", newline="") as f_match:
        writer = csv.DictWriter(f_match, fieldnames=DIM_MATCH_COLUMNS)
        writer.writeheader()

        for match_id in sorted(match_meta.keys()):
            mm = match_meta[match_id]
            teams = mm["teams"]
            team_items = sorted(teams.items(), key=lambda x: x[0])

            team_1_id, team_1_name = (team_items[0] if len(team_items) > 0 else ("", ""))
            team_2_id, team_2_name = (team_items[1] if len(team_items) > 1 else ("", ""))

            writer.writerow(
                {
                    "match_id": match_id,
                    "team_1_id": team_1_id,
                    "team_1_name": team_1_name,
                    "team_2_id": team_2_id,
                    "team_2_name": team_2_name,
                    "event_count": mm["event_count"],
                    "max_minute": mm["max_minute"],
                    "max_second": mm["max_second"],
                }
            )

    # Build dim_team
    if teams_path.exists():
        with teams_path.open("r", encoding="utf-8", newline="") as f_team_in:
            reader = csv.DictReader(f_team_in)
            for row in reader:
                team_id = as_int(row.get("team_id", ""))
                if team_id is None:
                    continue
                team_lookup[team_id] = (row.get("team_name") or "").strip()

    with dim_team_out.open("w", encoding="utf-8", newline="") as f_team_out:
        writer = csv.DictWriter(f_team_out, fieldnames=["team_id", "team_name"])
        writer.writeheader()
        for team_id in sorted(team_lookup.keys()):
            writer.writerow({"team_id": team_id, "team_name": team_lookup[team_id]})

    # Build dim_player
    if players_path.exists():
        with players_path.open("r", encoding="utf-8", newline="") as f_player_in:
            reader = csv.DictReader(f_player_in)
            for row in reader:
                player_id = as_int(row.get("player_id", ""))
                if player_id is None:
                    continue
                player_name = (row.get("player_name") or "").strip()
                team_id = as_int(row.get("team_id", ""))
                player_lookup[player_id] = (player_name, team_id)

    with dim_player_out.open("w", encoding="utf-8", newline="") as f_player_out:
        writer = csv.DictWriter(
            f_player_out, fieldnames=["player_id", "player_name", "team_id"]
        )
        writer.writeheader()
        for player_id in sorted(player_lookup.keys()):
            player_name, team_id = player_lookup[player_id]
            writer.writerow(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "team_id": "" if team_id is None else team_id,
                }
            )


def main() -> None:
    input_dir = Path("data_processed")
    output_dir = Path("data_processed")
    build_star_schema(input_dir=input_dir, output_dir=output_dir)
    print("Wrote star schema tables to data_processed/")


if __name__ == "__main__":
    main()
