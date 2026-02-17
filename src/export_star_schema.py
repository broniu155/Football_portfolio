
from __future__ import annotations

import argparse
import csv
import math
import tempfile
import time
from pathlib import Path

import pandas as pd


FACT_CANDIDATE_COLUMNS = [
    "event_id",
    "match_id",
    "event_index",
    "period",
    "minute",
    "second",
    "timestamp",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
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

FACT_EVENT_MINIMUM_COLUMNS = [
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "type_id",
    "period",
    "minute",
    "second",
    "location_x",
    "location_y",
    "play_pattern_id",
    "position_id",
    "under_pressure",
    "counterpress",
]

FACT_SHOT_BASE_COLUMNS = [
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

TIME_COLS = [
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

ZONE_COLS = [
    "zone_id",
    "zone_x_index",
    "zone_y_index",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "zone_label",
]

DIM_MATCH_COLS = [
    "match_id",
    "competition_id",
    "season_id",
    "match_date",
    "kick_off",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "stadium_name",
    "referee_name",
    "match_week",
    "match_status",
    "last_updated",
    "last_updated_360",
    "match_status_360",
    "match_label",
    "is_360_available",
]

TIME_KEY_COLUMN = "time_key"
ZONE_ID_COLUMN = "zone_id"
KNOWN_SHOT_TYPE_IDS = {16}


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def as_int(value: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def as_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
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
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f_in:
            line_count = sum(1 for _ in f_in)
        return max(0, line_count - 1)
    except OSError:
        return None


def print_progress(rows_processed: int, start_time: float, total_rows: int | None, label: str) -> None:
    elapsed = time.perf_counter() - start_time
    rate = rows_processed / elapsed if elapsed > 0 else 0.0
    if total_rows is not None and total_rows > 0:
        remaining = max(0, total_rows - rows_processed)
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        eta_text = format_duration(eta_seconds) if eta_seconds != float("inf") else "n/a"
        pct = min(100.0, (rows_processed / total_rows) * 100)
        print(
            f"[{label}] {rows_processed:,}/{total_rows:,} ({pct:5.1f}%) | "
            f"elapsed {format_duration(elapsed)} | {rate:,.0f} rows/s | ETA {eta_text}",
            flush=True,
        )
    else:
        print(
            f"[{label}] {rows_processed:,} rows | elapsed {format_duration(elapsed)} | {rate:,.0f} rows/s",
            flush=True,
        )


def upsert_id_name(store: dict[int, str], id_value: str, name_value: str | None) -> None:
    key = as_int(id_value)
    if key is None:
        return
    name = (name_value or "").strip()
    if key not in store:
        store[key] = name
    elif not store[key] and name:
        store[key] = name


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


def make_time_key(period: int | None, minute: int | None, second: int | None) -> int | None:
    if period is None or minute is None or second is None:
        return None
    return period * 100000 + minute * 100 + second


def minute_absolute(period: int, minute: int) -> int:
    return max(0, (period - 1) * 45 + minute)


def safe_reader(path: Path) -> tuple[csv.DictReader, object] | None:
    if not path.exists():
        warn(f"Missing source file: {path}. Skipping.")
        return None
    f_in = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(f_in)
    if not reader.fieldnames:
        f_in.close()
        warn(f"{path.name} has no header. Skipping.")
        return None
    return reader, f_in


def write_empty(path: Path, fieldnames: list[str]) -> int:
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
    return 0


def load_three_sixty_match_ids(three_sixty_path: Path, visible_area_path: Path) -> set[int]:
    match_ids: set[int] = set()
    for path in [visible_area_path, three_sixty_path]:
        reader_ctx = safe_reader(path)
        if reader_ctx is None:
            continue
        reader, f_in = reader_ctx
        try:
            for row in reader:
                match_id = as_int(row.get("match_id", ""))
                if match_id is not None:
                    match_ids.add(match_id)
        finally:
            f_in.close()
        if match_ids:
            break
    return match_ids


def build_dim_competition_and_season(
    competitions_path: Path,
    output_dir: Path,
    progress_every: int,
    estimate_total: bool,
) -> tuple[int, int]:
    dim_comp_path = output_dir / "dim_competition.csv"
    dim_season_path = output_dir / "dim_season.csv"

    reader_ctx = safe_reader(competitions_path)
    if reader_ctx is None:
        c = write_empty(
            dim_comp_path,
            [
                "competition_id",
                "competition_name",
                "country_name",
                "competition_gender",
                "competition_youth",
                "competition_international",
            ],
        )
        s = write_empty(dim_season_path, ["season_id", "season_name"])
        return c, s

    reader, f_in = reader_ctx
    competitions: dict[int, dict[str, object]] = {}
    seasons: dict[int, str] = {}

    total_rows = estimate_total_rows(competitions_path) if estimate_total else None
    start_time = time.perf_counter()
    rows = 0

    try:
        for row in reader:
            rows += 1
            if progress_every > 0 and rows % progress_every == 0:
                print_progress(rows, start_time, total_rows, "competitions")

            comp_id = as_int(row.get("competition_id", ""))
            if comp_id is not None and comp_id not in competitions:
                competitions[comp_id] = {
                    "competition_id": comp_id,
                    "competition_name": (row.get("competition_name") or "").strip(),
                    "country_name": (row.get("country_name") or "").strip(),
                    "competition_gender": (row.get("competition_gender") or "").strip(),
                    "competition_youth": (row.get("competition_youth") or "").strip(),
                    "competition_international": (row.get("competition_international") or "").strip(),
                }

            season_id = as_int(row.get("season_id", ""))
            if season_id is not None and season_id not in seasons:
                seasons[season_id] = (row.get("season_name") or "").strip()
    finally:
        f_in.close()

    if rows == 0:
        warn("competitions.csv has 0 data rows.")

    comp_written = 0
    with dim_comp_path.open("w", encoding="utf-8", newline="") as f_out:
        cols = [
            "competition_id",
            "competition_name",
            "country_name",
            "competition_gender",
            "competition_youth",
            "competition_international",
        ]
        writer = csv.DictWriter(f_out, fieldnames=cols)
        writer.writeheader()
        for comp_id in sorted(competitions.keys()):
            writer.writerow(competitions[comp_id])
            comp_written += 1

    season_written = 0
    with dim_season_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["season_id", "season_name"])
        writer.writeheader()
        for season_id in sorted(seasons.keys()):
            writer.writerow({"season_id": season_id, "season_name": seasons[season_id]})
            season_written += 1

    return comp_written, season_written

def load_matches_lookup(
    matches_path: Path,
    progress_every: int,
    estimate_total: bool,
) -> tuple[dict[int, dict[str, object]], int]:
    reader_ctx = safe_reader(matches_path)
    if reader_ctx is None:
        return {}, 0

    reader, f_in = reader_ctx
    match_lookup: dict[int, dict[str, object]] = {}
    total_rows = estimate_total_rows(matches_path) if estimate_total else None
    start_time = time.perf_counter()
    rows = 0

    try:
        for row in reader:
            rows += 1
            if progress_every > 0 and rows % progress_every == 0:
                print_progress(rows, start_time, total_rows, "matches")

            match_id = as_int(row.get("match_id", ""))
            if match_id is None:
                continue

            home_name = (row.get("home_team_name") or row.get("home_team") or "").strip()
            away_name = (row.get("away_team_name") or row.get("away_team") or "").strip()
            match_date = (row.get("match_date") or "").strip()
            label = f"{home_name} vs {away_name} ({match_date})" if home_name or away_name else ""

            match_lookup[match_id] = {
                "match_id": match_id,
                "competition_id": as_int(row.get("competition_id", "")),
                "season_id": as_int(row.get("season_id", "")),
                "match_date": match_date,
                "kick_off": (row.get("kick_off") or "").strip(),
                "home_team_id": as_int(row.get("home_team_id", "")),
                "away_team_id": as_int(row.get("away_team_id", "")),
                "home_team_name": home_name,
                "away_team_name": away_name,
                "home_score": as_int(row.get("home_score", "")),
                "away_score": as_int(row.get("away_score", "")),
                "stadium_name": (row.get("stadium_name") or "").strip(),
                "referee_name": (row.get("referee_name") or "").strip(),
                "match_week": as_int(row.get("match_week", "")),
                "match_status": (row.get("match_status") or "").strip(),
                "last_updated": (row.get("last_updated") or "").strip(),
                "last_updated_360": (row.get("last_updated_360") or "").strip(),
                "match_status_360": (row.get("match_status_360") or "").strip(),
                "match_label": label,
            }
    finally:
        f_in.close()

    if rows == 0:
        warn("matches.csv has 0 data rows.")

    return match_lookup, rows


def build_dim_match(
    matches_lookup: dict[int, dict[str, object]],
    three_sixty_match_ids: set[int],
    output_dir: Path,
) -> int:
    path = output_dir / "dim_match.csv"
    written = 0
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=DIM_MATCH_COLS)
        writer.writeheader()
        for match_id in sorted(matches_lookup.keys()):
            mm = matches_lookup[match_id]
            has_360_flag = str(mm.get("match_status_360", "")).strip() != "" or str(mm.get("last_updated_360", "")).strip() != ""
            has_360 = 1 if has_360_flag or match_id in three_sixty_match_ids else 0
            writer.writerow(
                {
                    "match_id": match_id,
                    "competition_id": "" if mm["competition_id"] is None else mm["competition_id"],
                    "season_id": "" if mm["season_id"] is None else mm["season_id"],
                    "match_date": mm["match_date"],
                    "kick_off": mm["kick_off"],
                    "home_team_id": "" if mm["home_team_id"] is None else mm["home_team_id"],
                    "away_team_id": "" if mm["away_team_id"] is None else mm["away_team_id"],
                    "home_score": "" if mm["home_score"] is None else mm["home_score"],
                    "away_score": "" if mm["away_score"] is None else mm["away_score"],
                    "stadium_name": mm["stadium_name"],
                    "referee_name": mm["referee_name"],
                    "match_week": "" if mm["match_week"] is None else mm["match_week"],
                    "match_status": mm["match_status"],
                    "last_updated": mm["last_updated"],
                    "last_updated_360": mm["last_updated_360"],
                    "match_status_360": mm["match_status_360"],
                    "match_label": mm["match_label"],
                    "is_360_available": has_360,
                }
            )
            written += 1
    return written


def load_team_player_lookups(
    input_dir: Path,
    matches_lookup: dict[int, dict[str, object]],
    progress_every: int,
    estimate_total: bool,
) -> tuple[dict[int, str], dict[int, tuple[str, int | None]], int]:
    teams_path = input_dir / "teams.csv"
    players_path = input_dir / "players.csv"
    lineups_path = input_dir / "lineups_players.csv"

    team_lookup: dict[int, str] = {}
    player_lookup: dict[int, tuple[str, int | None]] = {}

    reader_ctx = safe_reader(teams_path)
    if reader_ctx is not None:
        reader, f_in = reader_ctx
        try:
            for row in reader:
                team_id = as_int(row.get("team_id", ""))
                if team_id is None:
                    continue
                team_lookup[team_id] = (row.get("team_name") or "").strip()
        finally:
            f_in.close()

    reader_ctx = safe_reader(players_path)
    if reader_ctx is not None:
        reader, f_in = reader_ctx
        try:
            for row in reader:
                player_id = as_int(row.get("player_id", ""))
                if player_id is None:
                    continue
                player_lookup[player_id] = (
                    (row.get("player_name") or "").strip(),
                    as_int(row.get("team_id", "")),
                )
        finally:
            f_in.close()

    for mm in matches_lookup.values():
        home_id = mm.get("home_team_id")
        away_id = mm.get("away_team_id")
        if isinstance(home_id, int):
            team_lookup[home_id] = str(mm.get("home_team_name", "")).strip()
        if isinstance(away_id, int):
            team_lookup[away_id] = str(mm.get("away_team_name", "")).strip()

    lineups_rows = 0
    reader_ctx = safe_reader(lineups_path)
    if reader_ctx is not None:
        total_rows = estimate_total_rows(lineups_path) if estimate_total else None
        start_time = time.perf_counter()
        reader, f_in = reader_ctx
        try:
            for row in reader:
                lineups_rows += 1
                if progress_every > 0 and lineups_rows % progress_every == 0:
                    print_progress(lineups_rows, start_time, total_rows, "lineups_players")

                team_id = as_int(row.get("team_id", ""))
                if team_id is not None:
                    team_name = (row.get("team_name") or "").strip()
                    if team_id not in team_lookup:
                        team_lookup[team_id] = team_name
                    elif not team_lookup[team_id] and team_name:
                        team_lookup[team_id] = team_name

                player_id = as_int(row.get("player_id", ""))
                if player_id is None:
                    continue
                player_name = (row.get("player_name") or "").strip()
                existing = player_lookup.get(player_id)
                if existing is None:
                    player_lookup[player_id] = (player_name, team_id)
                else:
                    existing_name, existing_team = existing
                    resolved_name = existing_name if existing_name else player_name
                    resolved_team = existing_team if existing_team is not None else team_id
                    player_lookup[player_id] = (resolved_name, resolved_team)
        finally:
            f_in.close()

    return team_lookup, player_lookup, lineups_rows


def build_dim_team(output_dir: Path, team_lookup: dict[int, str]) -> int:
    path = output_dir / "dim_team.csv"
    written = 0
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["team_id", "team_name"])
        writer.writeheader()
        for team_id in sorted(team_lookup.keys()):
            writer.writerow({"team_id": team_id, "team_name": team_lookup[team_id]})
            written += 1
    return written


def build_dim_player(output_dir: Path, player_lookup: dict[int, tuple[str, int | None]]) -> int:
    path = output_dir / "dim_player.csv"
    written = 0
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["player_id", "player_name", "team_id"])
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
            written += 1
    return written


def build_dim_zone(output_dir: Path) -> int:
    path = output_dir / "dim_zone.csv"
    written = 0
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
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=ZONE_COLS)
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
                        "zone_label": f"Zone {zone_id} ({y_labels[zone_y]} {x_labels[zone_x]})",
                    }
                )
                written += 1
    return written


def build_dim_time(output_dir: Path, time_values: set[tuple[int, int, int]]) -> int:
    path = output_dir / "dim_time.csv"
    written = 0
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=TIME_COLS)
        writer.writeheader()
        for period, minute, second in sorted(time_values):
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
            written += 1
    return written


def write_id_name_dim(path: Path, id_col: str, name_col: str, values: dict[int, str]) -> int:
    written = 0
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[id_col, name_col])
        writer.writeheader()
        for key in sorted(values.keys()):
            writer.writerow({id_col: key, name_col: values[key]})
            written += 1
    return written

def build_fact_lineups_players(
    lineups_path: Path,
    output_dir: Path,
    progress_every: int,
    estimate_total: bool,
) -> int:
    out_path = output_dir / "fact_lineups_players.csv"
    cols = ["match_id", "team_id", "player_id", "position_id", "jersey_number", "starter", "from", "to"]

    reader_ctx = safe_reader(lineups_path)
    if reader_ctx is None:
        return write_empty(out_path, cols)

    total_rows = estimate_total_rows(lineups_path) if estimate_total else None
    start_time = time.perf_counter()
    rows_processed = 0
    rows_written = 0

    reader, f_in = reader_ctx
    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=cols)
        writer.writeheader()

        for row in reader:
            rows_processed += 1
            if progress_every > 0 and rows_processed % progress_every == 0:
                print_progress(rows_processed, start_time, total_rows, "fact_lineups_players")

            out_row = {}
            for col in cols:
                value = row.get(col, "")
                out_row[col] = "" if is_nested_like(value) else value
            writer.writerow(out_row)
            rows_written += 1

    f_in.close()

    if rows_processed == 0:
        warn("lineups_players.csv has 0 data rows. Wrote empty fact_lineups_players.csv.")

    return rows_written


def build_fact_three_sixty(
    three_sixty_path: Path,
    visible_area_path: Path,
    output_dir: Path,
    progress_every: int,
    estimate_total: bool,
) -> int:
    out_path = output_dir / "fact_three_sixty.csv"
    out_cols = ["match_id", "event_uuid", "visible_area_point_count", "has_visible_area"]

    reader_ctx = safe_reader(visible_area_path)
    source_path = visible_area_path
    source_label = "three_sixty_visible_area"
    if reader_ctx is None:
        reader_ctx = safe_reader(three_sixty_path)
        source_path = three_sixty_path
        source_label = "three_sixty"

    if reader_ctx is None:
        return write_empty(out_path, out_cols)

    reader, f_in = reader_ctx
    source_cols = set(reader.fieldnames or [])
    total_rows = estimate_total_rows(source_path) if estimate_total else None
    start_time = time.perf_counter()
    rows_processed = 0
    rows_written = 0

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_cols)
        writer.writeheader()

        if "visible_area_point_count" in source_cols:
            for row in reader:
                rows_processed += 1
                if progress_every > 0 and rows_processed % progress_every == 0:
                    print_progress(rows_processed, start_time, total_rows, "fact_three_sixty")

                point_count = as_int(row.get("visible_area_point_count", ""))
                has_visible = 1 if point_count is not None and point_count > 0 else 0
                writer.writerow(
                    {
                        "match_id": row.get("match_id", ""),
                        "event_uuid": row.get("event_uuid", ""),
                        "visible_area_point_count": "" if point_count is None else point_count,
                        "has_visible_area": has_visible,
                    }
                )
                rows_written += 1
        else:
            for row in reader:
                rows_processed += 1
                if progress_every > 0 and rows_processed % progress_every == 0:
                    print_progress(rows_processed, start_time, total_rows, "fact_three_sixty")

                point_count = as_int(row.get("visible_area_point_count", ""))
                has_visible = 1 if point_count is not None and point_count > 0 else 0
                writer.writerow(
                    {
                        "match_id": row.get("match_id", ""),
                        "event_uuid": row.get("event_uuid", ""),
                        "visible_area_point_count": "" if point_count is None else point_count,
                        "has_visible_area": has_visible,
                    }
                )
                rows_written += 1

    f_in.close()

    if rows_processed == 0:
        warn(f"{source_label}.csv has 0 data rows. Wrote empty fact_three_sixty.csv.")

    return rows_written


def build_fact_three_sixty_freeze_frames(
    freeze_frames_path: Path,
    output_dir: Path,
    progress_every: int,
    estimate_total: bool,
) -> int:
    out_path = output_dir / "fact_three_sixty_freeze_frames.csv"
    out_cols = [
        "match_id",
        "event_uuid",
        "player_id",
        "teammate",
        "actor",
        "keeper",
        "location_x",
        "location_y",
        "zone_id",
    ]

    reader_ctx = safe_reader(freeze_frames_path)
    if reader_ctx is None:
        return write_empty(out_path, out_cols)

    total_rows = estimate_total_rows(freeze_frames_path) if estimate_total else None
    start_time = time.perf_counter()
    rows_processed = 0
    rows_written = 0

    reader, f_in = reader_ctx
    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_cols)
        writer.writeheader()

        for row in reader:
            rows_processed += 1
            if progress_every > 0 and rows_processed % progress_every == 0:
                print_progress(rows_processed, start_time, total_rows, "fact_three_sixty_freeze_frames")

            x = row.get("location_x", "")
            y = row.get("location_y", "")
            zone_id = compute_zone_id(x, y)

            writer.writerow(
                {
                    "match_id": row.get("match_id", ""),
                    "event_uuid": row.get("event_uuid", ""),
                    "player_id": row.get("player_id", ""),
                    "teammate": row.get("teammate", ""),
                    "actor": row.get("actor", ""),
                    "keeper": row.get("keeper", ""),
                    "location_x": "" if is_nested_like(x) else x,
                    "location_y": "" if is_nested_like(y) else y,
                    "zone_id": "" if zone_id is None else zone_id,
                }
            )
            rows_written += 1

    f_in.close()

    if rows_processed == 0:
        warn("three_sixty_freeze_frames.csv has 0 data rows. Wrote empty fact_three_sixty_freeze_frames.csv.")

    return rows_written


def build_events_and_shots(
    events_path: Path,
    output_dir: Path,
    matches_lookup: dict[int, dict[str, object]],
    progress_every: int,
    estimate_total: bool,
) -> tuple[int, int, set[tuple[int, int, int]], dict[str, dict[int, str]]]:
    fact_events_path = output_dir / "fact_events.csv"
    fact_shots_path = output_dir / "fact_shots.csv"

    reader_ctx = safe_reader(events_path)
    if reader_ctx is None:
        write_empty(
            fact_events_path,
            FACT_EVENT_MINIMUM_COLUMNS + [ZONE_ID_COLUMN, TIME_KEY_COLUMN, "competition_id", "season_id"],
        )
        write_empty(
            fact_shots_path,
            FACT_SHOT_BASE_COLUMNS + ["shot_type_id", "shot_body_part_id", ZONE_ID_COLUMN, TIME_KEY_COLUMN],
        )
        return 0, 0, set(), {
            "event_type": {},
            "play_pattern": {},
            "position": {},
            "pass_outcome": {},
            "shot_outcome": {},
            "duel_type": {},
            "duel_outcome": {},
        }

    reader, f_in = reader_ctx
    source_cols = set(reader.fieldnames or [])

    event_small_dims = {
        "event_type": {},
        "play_pattern": {},
        "position": {},
        "pass_outcome": {},
        "shot_outcome": {},
        "duel_type": {},
        "duel_outcome": {},
    }

    fact_cols: list[str] = []
    for col in FACT_EVENT_MINIMUM_COLUMNS:
        if col in source_cols:
            fact_cols.append(col)
    for col in FACT_CANDIDATE_COLUMNS:
        if col in source_cols and col not in fact_cols:
            fact_cols.append(col)

    has_location = "location_x" in source_cols and "location_y" in source_cols
    if has_location:
        fact_cols.append(ZONE_ID_COLUMN)
    fact_cols.extend([TIME_KEY_COLUMN, "competition_id", "season_id"])

    shot_cols = [c for c in FACT_SHOT_BASE_COLUMNS if c in source_cols]
    if "shot_outcome_id" not in shot_cols:
        shot_cols.append("shot_outcome_id")
    if "shot_type_id" in source_cols:
        shot_cols.append("shot_type_id")
    if "shot_body_part_id" in source_cols:
        shot_cols.append("shot_body_part_id")
    if has_location:
        shot_cols.append(ZONE_ID_COLUMN)
    shot_cols.append(TIME_KEY_COLUMN)

    required = {"event_id", "match_id"}
    missing_required = sorted(required - set(fact_cols))
    if missing_required:
        f_in.close()
        warn(f"events.csv missing required columns {missing_required}. Skipping facts from events.")
        write_empty(fact_events_path, fact_cols)
        write_empty(fact_shots_path, shot_cols)
        return 0, 0, set(), event_small_dims

    total_rows = estimate_total_rows(events_path) if estimate_total else None
    start_time = time.perf_counter()
    rows_processed = 0
    events_written = 0
    shots_written = 0
    time_values: set[tuple[int, int, int]] = set()
    shot_type_ids = set(KNOWN_SHOT_TYPE_IDS)

    with (
        fact_events_path.open("w", encoding="utf-8", newline="") as f_events_out,
        fact_shots_path.open("w", encoding="utf-8", newline="") as f_shots_out,
    ):
        events_writer = csv.DictWriter(f_events_out, fieldnames=fact_cols)
        shots_writer = csv.DictWriter(f_shots_out, fieldnames=shot_cols)
        events_writer.writeheader()
        shots_writer.writeheader()

        for row in reader:
            rows_processed += 1
            if progress_every > 0 and rows_processed % progress_every == 0:
                print_progress(rows_processed, start_time, total_rows, "events")

            period = as_int(row.get("period", ""))
            minute = as_int(row.get("minute", ""))
            second = as_int(row.get("second", ""))
            time_key = make_time_key(period, minute, second)
            if period is not None and minute is not None and second is not None:
                time_values.add((period, minute, second))

            zone_id = None
            if has_location:
                zone_id = compute_zone_id(row.get("location_x", ""), row.get("location_y", ""))

            match_id = as_int(row.get("match_id", ""))
            match_meta = matches_lookup.get(match_id, {}) if match_id is not None else {}
            competition_id = match_meta.get("competition_id")
            season_id = match_meta.get("season_id")

            out_row: dict[str, object] = {}
            for col in fact_cols:
                if col == ZONE_ID_COLUMN:
                    out_row[col] = "" if zone_id is None else zone_id
                    continue
                if col == TIME_KEY_COLUMN:
                    out_row[col] = "" if time_key is None else time_key
                    continue
                if col == "competition_id":
                    out_row[col] = "" if competition_id is None else competition_id
                    continue
                if col == "season_id":
                    out_row[col] = "" if season_id is None else season_id
                    continue
                value = row.get(col, "")
                out_row[col] = "" if is_nested_like(value) else value
            events_writer.writerow(out_row)
            events_written += 1

            type_id = as_int(row.get("type_id", ""))
            type_name_norm = (row.get("type_name") or "").strip().lower()
            if type_name_norm == "shot" and type_id is not None:
                shot_type_ids.add(type_id)
            is_shot = type_name_norm == "shot" or (type_id is not None and type_id in shot_type_ids)

            if is_shot:
                shot_row: dict[str, object] = {}
                for col in shot_cols:
                    if col == ZONE_ID_COLUMN:
                        shot_row[col] = "" if zone_id is None else zone_id
                        continue
                    if col == TIME_KEY_COLUMN:
                        shot_row[col] = "" if time_key is None else time_key
                        continue
                    value = row.get(col, "")
                    shot_row[col] = "" if is_nested_like(value) else value
                shots_writer.writerow(shot_row)
                shots_written += 1

            upsert_id_name(event_small_dims["event_type"], row.get("type_id", ""), row.get("type_name", ""))
            upsert_id_name(
                event_small_dims["play_pattern"],
                row.get("play_pattern_id", ""),
                row.get("play_pattern_name", ""),
            )
            upsert_id_name(
                event_small_dims["position"],
                row.get("position_id", ""),
                row.get("position_name", ""),
            )
            upsert_id_name(
                event_small_dims["pass_outcome"],
                row.get("pass_outcome_id", ""),
                row.get("pass_outcome_name", ""),
            )
            upsert_id_name(
                event_small_dims["shot_outcome"],
                row.get("shot_outcome_id", ""),
                row.get("shot_outcome_name", ""),
            )
            upsert_id_name(
                event_small_dims["duel_type"],
                row.get("duel_type_id", ""),
                row.get("duel_type_name", ""),
            )
            upsert_id_name(
                event_small_dims["duel_outcome"],
                row.get("duel_outcome_id", ""),
                row.get("duel_outcome_name", ""),
            )

    f_in.close()

    if rows_processed == 0:
        warn("events.csv has 0 data rows. Skipping fact_events content and event-derived small dims.")

    return events_written, shots_written, time_values, event_small_dims


def build_small_dims(output_dir: Path, small_dims: dict[str, dict[int, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    mappings = [
        ("dim_event_type.csv", "type_id", "type_name", "event_type"),
        ("dim_play_pattern.csv", "play_pattern_id", "play_pattern_name", "play_pattern"),
        ("dim_position.csv", "position_id", "position_name", "position"),
        ("dim_pass_outcome.csv", "pass_outcome_id", "pass_outcome_name", "pass_outcome"),
        ("dim_shot_outcome.csv", "shot_outcome_id", "shot_outcome_name", "shot_outcome"),
        ("dim_duel_type.csv", "duel_type_id", "duel_type_name", "duel_type"),
        ("dim_duel_outcome.csv", "duel_outcome_id", "duel_outcome_name", "duel_outcome"),
    ]

    for filename, id_col, name_col, key in mappings:
        values = small_dims.get(key, {})
        if values:
            counts[filename] = write_id_name_dim(output_dir / filename, id_col, name_col, values)
        else:
            counts[filename] = write_empty(output_dir / filename, [id_col, name_col])
    return counts


def write_model_notes(output_dir: Path) -> int:
    path = output_dir / "model_notes.md"
    text = """# Power BI Star Schema Notes

## Tables and Grain
- `dim_competition`: one row per `competition_id`
- `dim_season`: one row per `season_id`
- `dim_match`: one row per `match_id`
- `dim_team`: one row per `team_id`
- `dim_player`: one row per `player_id`
- `dim_time`: one row per `(period, minute, second)` via `time_key`
- `dim_zone`: one row per static pitch zone (`6x5` grid)
- `fact_events`: one row per `event_id`
- `fact_shots`: one row per shot event
- `fact_lineups_players`: one row per `(match_id, team_id, player_id)` lineup entry
- `fact_three_sixty`: one row per `event_uuid`
- `fact_three_sixty_freeze_frames`: one row per `(match_id, event_uuid, player_id)` freeze-frame row

## Recommended Relationships
- `fact_events[match_id]` -> `dim_match[match_id]`
- `fact_events[player_id]` -> `dim_player[player_id]`
- `fact_events[team_id]` -> `dim_team[team_id]`
- `fact_events[time_key]` -> `dim_time[time_key]`
- `fact_events[zone_id]` -> `dim_zone[zone_id]`
- `fact_shots` uses the same relationship pattern as `fact_events`
- `fact_lineups_players[match_id]` -> `dim_match[match_id]`
- `fact_lineups_players[player_id]` -> `dim_player[player_id]`
- `fact_lineups_players[team_id]` -> `dim_team[team_id]`
- `fact_three_sixty[match_id]` -> `dim_match[match_id]`
- `fact_three_sixty_freeze_frames[player_id]` -> `dim_player[player_id]`
"""
    path.write_text(text, encoding="utf-8")
    return len(text)


def convert_csv_tables_to_parquet(csv_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        out_path = output_dir / f"{csv_path.stem}.parquet"
        df.to_parquet(out_path, index=False)


def _build_star_schema_csv(
    input_dir: Path,
    output_dir: Path,
    progress_every: int = 200_000,
    estimate_total: bool = False,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    competitions_path = input_dir / "competitions.csv"
    matches_path = input_dir / "matches.csv"
    events_path = input_dir / "events.csv"
    lineups_path = input_dir / "lineups_players.csv"
    three_sixty_path = input_dir / "three_sixty.csv"
    visible_area_path = input_dir / "three_sixty_visible_area.csv"
    freeze_frames_path = input_dir / "three_sixty_freeze_frames.csv"

    counts: dict[str, int] = {}

    comp_count, season_count = build_dim_competition_and_season(
        competitions_path, output_dir, progress_every, estimate_total
    )
    counts["dim_competition.csv"] = comp_count
    counts["dim_season.csv"] = season_count

    matches_lookup, _ = load_matches_lookup(matches_path, progress_every, estimate_total)
    three_sixty_match_ids = load_three_sixty_match_ids(three_sixty_path, visible_area_path)
    counts["dim_match.csv"] = build_dim_match(matches_lookup, three_sixty_match_ids, output_dir)

    team_lookup, player_lookup, _ = load_team_player_lookups(
        input_dir, matches_lookup, progress_every, estimate_total
    )
    counts["dim_team.csv"] = build_dim_team(output_dir, team_lookup)
    counts["dim_player.csv"] = build_dim_player(output_dir, player_lookup)

    counts["dim_zone.csv"] = build_dim_zone(output_dir)

    events_written, shots_written, time_values, small_dims = build_events_and_shots(
        events_path, output_dir, matches_lookup, progress_every, estimate_total
    )
    counts["fact_events.csv"] = events_written
    counts["fact_shots.csv"] = shots_written

    counts["dim_time.csv"] = build_dim_time(output_dir, time_values)

    counts["fact_lineups_players.csv"] = build_fact_lineups_players(
        lineups_path, output_dir, progress_every, estimate_total
    )
    counts["fact_three_sixty.csv"] = build_fact_three_sixty(
        three_sixty_path, visible_area_path, output_dir, progress_every, estimate_total
    )
    counts["fact_three_sixty_freeze_frames.csv"] = build_fact_three_sixty_freeze_frames(
        freeze_frames_path, output_dir, progress_every, estimate_total
    )

    small_dim_counts = build_small_dims(output_dir, small_dims)
    counts.update(small_dim_counts)

    write_model_notes(output_dir)
    return counts


def build_star_schema(
    input_dir: Path,
    output_dir: Path,
    progress_every: int = 200_000,
    estimate_total: bool = False,
    output_format: str = "parquet",
) -> dict[str, int]:
    if output_format not in {"csv", "parquet"}:
        raise ValueError("output_format must be 'csv' or 'parquet'")

    if output_format == "csv":
        return _build_star_schema_csv(
            input_dir=input_dir,
            output_dir=output_dir,
            progress_every=progress_every,
            estimate_total=estimate_total,
        )

    with tempfile.TemporaryDirectory(prefix="star_schema_csv_") as temp_dir:
        temp_output_dir = Path(temp_dir)
        counts = _build_star_schema_csv(
            input_dir=input_dir,
            output_dir=temp_output_dir,
            progress_every=progress_every,
            estimate_total=estimate_total,
        )
        convert_csv_tables_to_parquet(temp_output_dir, output_dir)
        write_model_notes(output_dir)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export StatsBomb star-schema CSVs for Power BI.")
    parser.add_argument("--input-dir", type=Path, default=Path("data_processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_model"))
    parser.add_argument("--progress-every", type=int, default=200000)
    parser.add_argument("--estimate-total", action="store_true", default=False)
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_star_schema(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
        estimate_total=args.estimate_total,
        output_format=args.format,
    )

    print("\nBuild summary (rows written):")
    for table in sorted(counts.keys()):
        print(f"- {table}: {counts[table]:,}")
    print(f"\nWrote star schema tables to {args.output_dir}/ ({args.format})")


if __name__ == "__main__":
    main()
