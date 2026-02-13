from __future__ import annotations

import csv
import math
import time
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
    "play_pattern_id",
    "play_pattern_name",
    "position_id",
    "position_name",
    "pass_outcome_id",
    "pass_outcome_name",
    "shot_outcome_id",
    "shot_outcome_name",
    "duel_type_id",
    "duel_type_name",
    "duel_outcome_id",
    "duel_outcome_name",
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
    "carry_end_location_x",
    "carry_end_location_y",
    "shot_end_location_x",
    "shot_end_location_y",
    "shot_statsbomb_xg",
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

SHOT_FACT_BASE_COLUMNS = [
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "minute",
    "second",
    "period",
    "location_x",
    "location_y",
    "shot_end_location_x",
    "shot_end_location_y",
    "shot_statsbomb_xg",
    "shot_outcome_id",
    "under_pressure",
]

TIME_KEY_COLUMN = "time_key"
ZONE_ID_COLUMN = "zone_id"
KNOWN_SHOT_TYPE_IDS = {16}


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


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def estimate_total_rows(csv_path: Path) -> int | None:
    """Estimate total data rows using a fast line count pass (excludes header)."""
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f_in:
            line_count = sum(1 for _ in f_in)
        return max(0, line_count - 1)
    except OSError:
        return None


def print_progress(rows_processed: int, start_time: float, total_rows: int | None) -> None:
    elapsed = time.perf_counter() - start_time
    rate = rows_processed / elapsed if elapsed > 0 else 0.0

    if total_rows is not None and total_rows > 0:
        remaining = max(0, total_rows - rows_processed)
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        eta_text = format_duration(eta_seconds) if eta_seconds != float("inf") else "n/a"
        pct = min(100.0, (rows_processed / total_rows) * 100)
        print(
            f"Processed {rows_processed:,}/{total_rows:,} rows ({pct:5.1f}%) | "
            f"elapsed {format_duration(elapsed)} | {rate:,.0f} rows/s | ETA {eta_text}",
            flush=True,
        )
    else:
        print(
            f"Processed {rows_processed:,} rows | elapsed {format_duration(elapsed)} | {rate:,.0f} rows/s",
            flush=True,
        )


def upsert_id_name(
    store: dict[int, str], id_value: str, name_value: str | None, fallback_name: str = ""
) -> None:
    key = as_int(id_value)
    if key is None:
        return
    name = (name_value or "").strip()
    if key not in store:
        store[key] = name if name else fallback_name
    elif not store[key] and name:
        store[key] = name


def write_id_name_dim(path: Path, id_col: str, name_col: str, values: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[id_col, name_col])
        writer.writeheader()
        for key in sorted(values.keys()):
            writer.writerow({id_col: key, name_col: values[key]})


def write_single_col_dim(path: Path, col: str, values: set[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[col])
        writer.writeheader()
        for value in sorted(values):
            writer.writerow({col: value})


def compute_zone_id(location_x: str, location_y: str) -> int | None:
    x = as_float(location_x)
    y = as_float(location_y)
    if x is None or y is None:
        return None
    if x < 0 or y < 0 or x > 120 or y > 80:
        return None
    zone_x = min(5, int(math.floor(x / 20)))
    zone_y = min(4, int(math.floor(y / 16)))
    return zone_y * 6 + zone_x


def build_zone_label(zone_x: int, zone_y: int) -> str:
    x_labels = [
        "Defensive Sixth",
        "Build-Up Sixth",
        "Middle Sixth",
        "Attacking Sixth",
        "Final Third",
        "Box Sixth",
    ]
    y_labels = [
        "Left Wing",
        "Left Half-Space",
        "Central",
        "Right Half-Space",
        "Right Wing",
    ]
    return f"{y_labels[zone_y]} {x_labels[zone_x]}"


def write_dim_zone(path: Path) -> None:
    zone_cols = [
        "zone_id",
        "zone_x_index",
        "zone_y_index",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "zone_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=zone_cols)
        writer.writeheader()
        for zone_y in range(5):
            for zone_x in range(6):
                zone_id = zone_y * 6 + zone_x
                x_min = zone_x * 20
                y_min = zone_y * 16
                writer.writerow(
                    {
                        "zone_id": zone_id,
                        "zone_x_index": zone_x,
                        "zone_y_index": zone_y,
                        "x_min": x_min,
                        "x_max": x_min + 20,
                        "y_min": y_min,
                        "y_max": y_min + 16,
                        "zone_label": f"Zone {zone_id} ({build_zone_label(zone_x, zone_y)})",
                    }
                )


def make_time_key(period: int | None, minute: int | None, second: int | None) -> int | None:
    if period is None or minute is None or second is None:
        return None
    return period * 100000 + minute * 100 + second


def minute_absolute(period: int, minute: int) -> int:
    return max(0, (period - 1) * 45 + minute)


def write_dim_time(path: Path, values: set[tuple[int, int, int]]) -> None:
    time_cols = [
        "time_key",
        "period",
        "minute",
        "second",
        "minute_absolute",
        "time_bucket_5min",
        "time_bucket_15min",
        "is_first_half",
        "is_second_half",
        "is_extra_time",
    ]
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=time_cols)
        writer.writeheader()
        for period, minute, second in sorted(values):
            minute_abs = minute_absolute(period, minute)
            writer.writerow(
                {
                    "time_key": make_time_key(period, minute, second),
                    "period": period,
                    "minute": minute,
                    "second": second,
                    "minute_absolute": minute_abs,
                    "time_bucket_5min": (minute_abs // 5) * 5,
                    "time_bucket_15min": (minute_abs // 15) * 15,
                    "is_first_half": 1 if period == 1 else 0,
                    "is_second_half": 1 if period == 2 else 0,
                    "is_extra_time": 1 if period >= 3 else 0,
                }
            )


def build_star_schema(
    input_dir: Path,
    output_dir: Path,
    progress_every: int = 50_000,
    estimate_total: bool = True,
) -> None:
    events_path = input_dir / "events.csv"
    teams_path = input_dir / "teams.csv"
    players_path = input_dir / "players.csv"

    if not events_path.exists():
        raise FileNotFoundError(f"Missing source file: {events_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fact_out = output_dir / "fact_events.csv"
    fact_shots_out = output_dir / "fact_shots.csv"
    dim_match_out = output_dir / "dim_match.csv"
    dim_team_out = output_dir / "dim_team.csv"
    dim_player_out = output_dir / "dim_player.csv"
    dim_zone_out = output_dir / "dim_zone.csv"
    dim_time_out = output_dir / "dim_time.csv"

    match_meta: dict[int, dict[str, object]] = {}
    team_lookup: dict[int, str] = {}
    player_lookup: dict[int, tuple[str, int | None]] = {}

    event_type_lookup: dict[int, str] = {}
    period_values: set[int] = set()
    play_pattern_lookup: dict[int, str] = {}
    position_lookup: dict[int, str] = {}
    pass_outcome_lookup: dict[int, str] = {}
    shot_outcome_lookup: dict[int, str] = {}
    duel_type_lookup: dict[int, str] = {}
    duel_outcome_lookup: dict[int, str] = {}
    time_values: set[tuple[int, int, int]] = set()
    shot_type_ids = set(KNOWN_SHOT_TYPE_IDS)

    total_rows = estimate_total_rows(events_path) if estimate_total else None
    if total_rows is not None:
        print(f"Scanning events.csv with progress updates every {progress_every:,} rows (total rows: {total_rows:,})")
    else:
        print(f"Scanning events.csv with progress updates every {progress_every:,} rows")

    start_time = time.perf_counter()
    rows_processed = 0

    with (
        events_path.open("r", encoding="utf-8", newline="") as f_in,
        fact_out.open("w", encoding="utf-8", newline="") as f_out,
        fact_shots_out.open("w", encoding="utf-8", newline="") as f_shot_out,
    ):
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("events.csv has no header")

        source_cols = set(reader.fieldnames)
        fact_cols = [c for c in FACT_CANDIDATE_COLUMNS if c in source_cols]

        shot_fact_cols = SHOT_FACT_BASE_COLUMNS.copy()
        if "shot_body_part_id" in source_cols:
            shot_fact_cols.append("shot_body_part_id")
        if "shot_type_id" in source_cols:
            shot_fact_cols.append("shot_type_id")

        has_location = "location_x" in source_cols and "location_y" in source_cols
        if has_location:
            fact_cols.append(ZONE_ID_COLUMN)
            shot_fact_cols.append(ZONE_ID_COLUMN)

        # Add a stable surrogate key to align both fact tables with dim_time.
        fact_cols.append(TIME_KEY_COLUMN)
        shot_fact_cols.append(TIME_KEY_COLUMN)

        required_fact = {"event_id", "match_id", "team_id", "player_id"}
        missing_req = sorted(required_fact - set(fact_cols))
        if missing_req:
            raise ValueError(f"Missing required fact columns: {missing_req}")

        writer = csv.DictWriter(f_out, fieldnames=fact_cols)
        writer.writeheader()
        shot_writer = csv.DictWriter(f_shot_out, fieldnames=shot_fact_cols)
        shot_writer.writeheader()

        has_type = "type_id" in source_cols
        has_period = "period" in source_cols
        has_play_pattern = "play_pattern_id" in source_cols
        has_position = "position_id" in source_cols
        has_pass_outcome = "pass_outcome_id" in source_cols
        has_shot_outcome = "shot_outcome_id" in source_cols
        has_duel_type = "duel_type_id" in source_cols
        has_duel_outcome = "duel_outcome_id" in source_cols
        has_type_name = "type_name" in source_cols
        has_type_id = "type_id" in source_cols

        for row in reader:
            period = as_int(row.get("period", ""))
            minute = as_int(row.get("minute", ""))
            second = as_int(row.get("second", ""))
            event_time_key = make_time_key(period, minute, second)
            if period is not None and minute is not None and second is not None:
                time_values.add((period, minute, second))

            zone_id: int | None = None
            if has_location:
                zone_id = compute_zone_id(row.get("location_x", ""), row.get("location_y", ""))

            flat_row: dict[str, str] = {}
            for col in fact_cols:
                if col == ZONE_ID_COLUMN:
                    flat_row[col] = "" if zone_id is None else str(zone_id)
                    continue
                if col == TIME_KEY_COLUMN:
                    flat_row[col] = "" if event_time_key is None else str(event_time_key)
                    continue
                val = row.get(col, "")
                flat_row[col] = "" if is_nested_like(val) else val
            writer.writerow(flat_row)

            type_id = as_int(row.get("type_id", ""))
            type_name_norm = (row.get("type_name") or "").strip().lower()
            if type_name_norm == "shot" and type_id is not None:
                shot_type_ids.add(type_id)

            # A row is a shot if type_name says Shot or type_id is recognized as shot.
            is_shot = (has_type_name and type_name_norm == "shot") or (
                has_type_id and type_id is not None and type_id in shot_type_ids
            )
            if is_shot:
                shot_row: dict[str, str] = {}
                for col in shot_fact_cols:
                    if col == ZONE_ID_COLUMN:
                        shot_row[col] = "" if zone_id is None else str(zone_id)
                        continue
                    if col == TIME_KEY_COLUMN:
                        shot_row[col] = "" if event_time_key is None else str(event_time_key)
                        continue
                    val = row.get(col, "")
                    shot_row[col] = "" if is_nested_like(val) else val
                shot_writer.writerow(shot_row)

            rows_processed += 1
            if progress_every > 0 and rows_processed % progress_every == 0:
                print_progress(rows_processed, start_time, total_rows)

            match_id = as_int(row.get("match_id", ""))
            team_id = as_int(row.get("team_id", ""))
            player_id = as_int(row.get("player_id", ""))
            team_name = (row.get("team_name") or "").strip()
            player_name = (row.get("player_name") or "").strip()

            if match_id is not None:
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

            if has_type:
                upsert_id_name(
                    event_type_lookup,
                    row.get("type_id", ""),
                    row.get("type_name", ""),
                )
            if has_period and period is not None:
                period_values.add(period)
            if has_play_pattern:
                upsert_id_name(
                    play_pattern_lookup,
                    row.get("play_pattern_id", ""),
                    row.get("play_pattern_name", ""),
                )
            if has_position:
                upsert_id_name(
                    position_lookup,
                    row.get("position_id", ""),
                    row.get("position_name", ""),
                )
            if has_pass_outcome:
                upsert_id_name(
                    pass_outcome_lookup,
                    row.get("pass_outcome_id", ""),
                    row.get("pass_outcome_name", ""),
                )
            if has_shot_outcome:
                upsert_id_name(
                    shot_outcome_lookup,
                    row.get("shot_outcome_id", ""),
                    row.get("shot_outcome_name", ""),
                )
            if has_duel_type:
                upsert_id_name(
                    duel_type_lookup,
                    row.get("duel_type_id", ""),
                    row.get("duel_type_name", ""),
                )
            if has_duel_outcome:
                upsert_id_name(
                    duel_outcome_lookup,
                    row.get("duel_outcome_id", ""),
                    row.get("duel_outcome_name", ""),
                )

    print_progress(rows_processed, start_time, total_rows)

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

    # Static pitch zone dimension (6x5 on a 120x80 pitch).
    write_dim_zone(dim_zone_out)

    # Build dim_time from unique streamed event timestamps.
    if time_values:
        write_dim_time(dim_time_out, time_values)

    # Build optional small dimensions. Files are written only when source columns exist.
    if event_type_lookup:
        write_id_name_dim(output_dir / "dim_event_type.csv", "type_id", "type_name", event_type_lookup)
    if period_values:
        write_single_col_dim(output_dir / "dim_period.csv", "period", period_values)
    if play_pattern_lookup:
        write_id_name_dim(
            output_dir / "dim_play_pattern.csv",
            "play_pattern_id",
            "play_pattern_name",
            play_pattern_lookup,
        )
    if position_lookup:
        write_id_name_dim(output_dir / "dim_position.csv", "position_id", "position_name", position_lookup)
    if pass_outcome_lookup:
        write_id_name_dim(
            output_dir / "dim_pass_outcome.csv",
            "pass_outcome_id",
            "pass_outcome_name",
            pass_outcome_lookup,
        )
    if shot_outcome_lookup:
        write_id_name_dim(
            output_dir / "dim_shot_outcome.csv",
            "shot_outcome_id",
            "shot_outcome_name",
            shot_outcome_lookup,
        )
    if duel_type_lookup:
        write_id_name_dim(
            output_dir / "dim_duel_type.csv",
            "duel_type_id",
            "duel_type_name",
            duel_type_lookup,
        )
    if duel_outcome_lookup:
        write_id_name_dim(
            output_dir / "dim_duel_outcome.csv",
            "duel_outcome_id",
            "duel_outcome_name",
            duel_outcome_lookup,
        )


def main() -> None:
    input_dir = Path("data_processed")
    output_dir = Path("data_processed")
    build_star_schema(input_dir=input_dir, output_dir=output_dir)
    print("Wrote star schema tables to data_processed/")


if __name__ == "__main__":
    main()
