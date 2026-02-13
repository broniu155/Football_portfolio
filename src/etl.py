
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable


REQUIRED_COLUMNS = {
    "events": ["event_id", "match_id", "team_id", "player_id", "type_id", "type_name", "minute", "second"],
    "competitions": [
        "competition_id",
        "competition_name",
        "country_name",
        "competition_gender",
        "competition_youth",
        "season_id",
        "season_name",
    ],
    "matches": [
        "match_id",
        "competition_id",
        "season_id",
        "match_date",
        "kick_off",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
    ],
    "lineups_players": [
        "match_id",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "jersey_number",
        "position_id",
        "position_name",
        "from",
        "to",
        "starter",
    ],
    "three_sixty": ["match_id", "event_uuid"],
    "three_sixty_freeze_frames": ["match_id", "event_uuid"],
    "three_sixty_visible_area": ["match_id", "event_uuid", "visible_area", "visible_area_point_count"],
    "teams": ["team_id", "team_name"],
    "players": ["player_id", "player_name", "team_id"],
}

KEY_NULL_COLUMNS = {
    "events": ["match_id", "team_id", "player_id"],
    "competitions": ["competition_id", "season_id"],
    "matches": ["match_id", "competition_id", "season_id", "home_team_id", "away_team_id"],
    "lineups_players": ["match_id", "team_id", "player_id"],
    "three_sixty": ["match_id", "event_uuid"],
    "three_sixty_freeze_frames": ["match_id", "event_uuid", "player_id"],
    "three_sixty_visible_area": ["match_id", "event_uuid"],
    "teams": ["team_id"],
    "players": ["player_id", "team_id"],
}

COORDINATE_KEYS = {
    "location",
    "pass_end_location",
    "carry_end_location",
    "shot_end_location",
}

PROCESSORS = ["events", "competitions", "matches", "lineups", "three-sixty"]


class ProgressPrinter:
    def __init__(
        self,
        total_files: int,
        phase: str,
        log_every_files: int,
        log_every_rows: int,
    ) -> None:
        self.total_files = max(total_files, 1)
        self.phase = phase
        self.log_every_files = max(1, log_every_files)
        self.log_every_rows = max(1, log_every_rows)
        self.start = time.perf_counter()
        self.last_file_log = 0
        self.last_row_log = 0

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = int(seconds)
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def maybe_log(self, file_count: int, row_count: int, force: bool = False) -> None:
        should_log = force
        if file_count - self.last_file_log >= self.log_every_files:
            should_log = True
        if row_count - self.last_row_log >= self.log_every_rows:
            should_log = True
        if not should_log:
            return

        elapsed = time.perf_counter() - self.start
        rate = row_count / elapsed if elapsed > 0 else 0.0
        print(
            f"[{self.phase}] Processed files: {file_count}/{self.total_files} "
            f"| rows: {row_count:,} "
            f"| rate: {rate:,.0f} rows/s "
            f"| elapsed: {self._format_elapsed(elapsed)}"
        )
        self.last_file_log = file_count
        self.last_row_log = row_count


class QualityTracker:
    def __init__(self, key_columns: Iterable[str], sample_limit: int | None = None) -> None:
        self.key_columns = list(key_columns)
        self.sample_limit = sample_limit
        self.row_count = 0
        self.sampled_rows = 0
        self.null_counts = {col: 0 for col in self.key_columns}

    @staticmethod
    def _is_null(value: Any) -> bool:
        return value in (None, "")

    def update(self, row: dict[str, Any]) -> None:
        self.row_count += 1
        if self.sample_limit is not None and self.sampled_rows >= self.sample_limit:
            return
        self.sampled_rows += 1
        for col in self.key_columns:
            if self._is_null(row.get(col, "")):
                self.null_counts[col] += 1

    def null_rates(self, available_columns: set[str]) -> dict[str, str]:
        rates: dict[str, str] = {}
        for col in self.key_columns:
            if col not in available_columns:
                rates[col] = "column missing"
            elif self.sampled_rows == 0:
                rates[col] = "not computed"
            else:
                rates[col] = f"{(self.null_counts[col] / self.sampled_rows) * 100:.2f}%"
        return rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process StatsBomb JSON folders to flat CSV outputs with data quality reporting."
    )
    parser.add_argument("--input-dir", default="data_raw", help="Input root directory.")
    parser.add_argument("--output-dir", default="data_processed", help="Output CSV directory.")
    parser.add_argument("--report-path", default="reports/data_quality.md", help="Markdown data quality report path.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit of JSON files per processor.")
    parser.add_argument("--log-every-files", type=int, default=50, help="Progress log interval in files.")
    parser.add_argument("--log-every-rows", type=int, default=100000, help="Progress log interval in rows.")
    parser.add_argument(
        "--report-sample-rows",
        type=int,
        default=None,
        help="Optional per-table sample row cap for null-rate stats.",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Comma-separated processors to run: events,competitions,matches,lineups,three-sixty",
    )
    parser.add_argument("--exclude", default=None, help="Comma-separated processors to skip.")
    return parser.parse_args()


def parse_processor_list(raw: str | None) -> set[str]:
    if raw is None or raw.strip() == "":
        return set()
    values = {x.strip() for x in raw.split(",") if x.strip()}
    invalid = sorted(values - set(PROCESSORS))
    if invalid:
        raise ValueError(f"Unknown processor(s): {invalid}. Valid processors: {PROCESSORS}")
    return values


def should_run_processor(name: str, include: set[str], exclude: set[str]) -> bool:
    if include and name not in include:
        return False
    if name in exclude:
        return False
    return True


def maybe_limit(files: list[Path], max_files: int | None) -> list[Path]:
    if max_files is not None and max_files > 0:
        return files[:max_files]
    return files


def resolve_events_files(input_dir: Path, max_files: int | None) -> list[Path]:
    events_dir = input_dir / "events"
    files = sorted(events_dir.glob("*.json")) if events_dir.exists() else sorted(input_dir.glob("*.json"))
    return maybe_limit(files, max_files)


def resolve_competitions_files(input_dir: Path, max_files: int | None) -> list[Path]:
    files: list[Path] = []
    direct = input_dir / "competitions.json"
    if direct.exists():
        files.append(direct)
    folder = input_dir / "competitions"
    if folder.exists():
        files.extend(sorted(folder.rglob("*.json")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for file in files:
        if file not in seen:
            seen.add(file)
            unique.append(file)
    return maybe_limit(unique, max_files)


def resolve_recursive_json(input_dir: Path, folder: str, max_files: int | None) -> list[Path]:
    target = input_dir / folder
    files = sorted(target.rglob("*.json")) if target.exists() else []
    return maybe_limit(files, max_files)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def parse_numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def is_numeric_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, (int, float)) for v in value)


def flatten_json(source: dict[str, Any], parent_key: str = "", split_coordinate_lists: bool = True) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in source.items():
        out_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(flatten_json(value, out_key, split_coordinate_lists))
            continue
        if split_coordinate_lists and is_numeric_list(value) and len(value) in (2, 3):
            is_coordinate = out_key in COORDINATE_KEYS or out_key.endswith("_location")
            flat[out_key] = json.dumps(value)
            if is_coordinate:
                axis_names = ["x", "y", "z"]
                for idx, item in enumerate(value):
                    suffix = axis_names[idx] if idx < len(axis_names) else f"coord_{idx + 1}"
                    flat[f"{out_key}_{suffix}"] = item
            continue
        if isinstance(value, list):
            flat[out_key] = json.dumps(value)
            continue
        flat[out_key] = value
    return flat


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_team_player_from_row(
    row: dict[str, Any], teams: dict[int, str], players: dict[int, tuple[str, int | None]]
) -> None:
    team_id = safe_int(row.get("team_id"))
    if team_id is not None:
        teams[team_id] = str(row.get("team_name") or "")
    player_id = safe_int(row.get("player_id"))
    if player_id is not None:
        players[player_id] = (str(row.get("player_name") or ""), team_id)


def finalize_table_quality(
    table_name: str, row_count: int, available_columns: set[str], tracker: QualityTracker
) -> dict[str, Any]:
    required = REQUIRED_COLUMNS.get(table_name, [])
    missing = [c for c in required if c not in available_columns]
    return {
        "table": table_name,
        "row_count": row_count,
        "available_columns": available_columns,
        "missing_required": missing,
        "null_rates": tracker.null_rates(available_columns),
        "sampled_rows": tracker.sampled_rows,
    }


def normalize_event_record(record: dict[str, Any], match_id: int) -> dict[str, Any]:
    row = flatten_json(record)
    if "id" in row and "event_id" not in row:
        row["event_id"] = row.pop("id")
    if "index" in row and "event_index" not in row:
        row["event_index"] = row.pop("index")
    row["match_id"] = match_id
    for col in ["match_id", "team_id", "player_id", "type_id", "minute", "second"]:
        row[col] = safe_int(row.get(col))
    return row

def process_events(
    input_dir: Path,
    output_dir: Path,
    max_files: int | None,
    log_every_files: int,
    log_every_rows: int,
    sample_rows: int | None,
    teams: dict[int, str],
    players: dict[int, tuple[str, int | None]],
) -> dict[str, Any] | None:
    files = resolve_events_files(input_dir, max_files)
    if not files:
        print("[events] skipping folder events (no files found)")
        return None

    all_columns: set[str] = set()
    unique_matches: set[int] = set()
    rows_pass1 = 0

    logger1 = ProgressPrinter(len(files), "events pass 1/2", log_every_files, log_every_rows)
    for idx, path in enumerate(files, start=1):
        match_id = parse_numeric_stem(path)
        if match_id is None:
            raise ValueError(f"Cannot parse match_id from filename: {path.name}")
        unique_matches.add(match_id)
        for record in load_json_records(path):
            row = normalize_event_record(record, match_id)
            all_columns.update(row.keys())
            rows_pass1 += 1
            update_team_player_from_row(row, teams, players)
        logger1.maybe_log(idx, rows_pass1)
    logger1.maybe_log(len(files), rows_pass1, force=True)

    missing_required = [c for c in REQUIRED_COLUMNS["events"] if c not in all_columns]
    if missing_required:
        raise ValueError(f"[events] missing required columns after flattening: {missing_required}")

    preferred = [
        "event_id",
        "match_id",
        "event_index",
        "period",
        "timestamp",
        "minute",
        "second",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "type_id",
        "type_name",
        "location",
        "location_x",
        "location_y",
        "location_z",
        "pass_end_location",
        "pass_end_location_x",
        "pass_end_location_y",
        "pass_end_location_z",
    ]
    ordered = [c for c in preferred if c in all_columns] + sorted(c for c in all_columns if c not in preferred)

    output_path = output_dir / "events.csv"
    tracker = QualityTracker(KEY_NULL_COLUMNS["events"], sample_rows)
    rows_written = 0
    logger2 = ProgressPrinter(len(files), "events pass 2/2", log_every_files, log_every_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=ordered)
        writer.writeheader()
        for idx, path in enumerate(files, start=1):
            match_id = int(path.stem)
            for record in load_json_records(path):
                row = normalize_event_record(record, match_id)
                out = {col: row.get(col, "") for col in ordered}
                writer.writerow(out)
                tracker.update(out)
                rows_written += 1
            logger2.maybe_log(idx, rows_written)
    logger2.maybe_log(len(files), rows_written, force=True)

    return {
        **finalize_table_quality("events", rows_written, set(ordered), tracker),
        "output": str(output_path),
        "files_processed": len(files),
        "unique_matches": len(unique_matches),
    }


def normalize_competition_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "season_id" not in out:
        out["season_id"] = out.get("season_season_id")
    if "season_name" not in out:
        out["season_name"] = out.get("season_season_name")
    if "competition_id" not in out:
        out["competition_id"] = out.get("competition_competition_id")
    if "competition_name" not in out:
        out["competition_name"] = out.get("competition_competition_name")
    if "country_name" not in out:
        out["country_name"] = out.get("country_name") or out.get("competition_country_name")
    return out


def process_competitions(
    input_dir: Path,
    output_dir: Path,
    max_files: int | None,
    log_every_files: int,
    log_every_rows: int,
    sample_rows: int | None,
) -> dict[str, Any] | None:
    files = resolve_competitions_files(input_dir, max_files)
    if not files:
        print("[competitions] skipping folder competitions (no files found)")
        return None

    rows: list[dict[str, Any]] = []
    columns: set[str] = set()
    tracker = QualityTracker(KEY_NULL_COLUMNS["competitions"], sample_rows)
    row_count = 0
    logger = ProgressPrinter(len(files), "competitions", log_every_files, log_every_rows)

    for idx, path in enumerate(files, start=1):
        for record in load_json_records(path):
            row = normalize_competition_row(flatten_json(record))
            rows.append(row)
            columns.update(row.keys())
            tracker.update(row)
            row_count += 1
        logger.maybe_log(idx, row_count)
    logger.maybe_log(len(files), row_count, force=True)

    preferred = REQUIRED_COLUMNS["competitions"]
    ordered = [c for c in preferred if c in columns] + sorted(c for c in columns if c not in preferred)
    output_path = output_dir / "competitions.csv"
    write_rows(output_path, ordered, [{c: row.get(c, "") for c in ordered} for row in rows])

    return {
        **finalize_table_quality("competitions", row_count, set(ordered), tracker),
        "output": str(output_path),
        "files_processed": len(files),
    }


def normalize_match_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["match_id"] = safe_int(out.get("match_id") or out.get("match_match_id"))
    out["competition_id"] = safe_int(out.get("competition_id") or out.get("competition_competition_id"))
    out["season_id"] = safe_int(out.get("season_id") or out.get("season_season_id"))
    out["home_team_id"] = safe_int(out.get("home_team_id") or out.get("home_team_home_team_id"))
    out["home_team_name"] = out.get("home_team_name") or out.get("home_team_home_team_name") or ""
    out["away_team_id"] = safe_int(out.get("away_team_id") or out.get("away_team_away_team_id"))
    out["away_team_name"] = out.get("away_team_name") or out.get("away_team_away_team_name") or ""
    return out


def process_matches(
    input_dir: Path,
    output_dir: Path,
    max_files: int | None,
    log_every_files: int,
    log_every_rows: int,
    sample_rows: int | None,
    teams: dict[int, str],
) -> dict[str, Any] | None:
    files = resolve_recursive_json(input_dir, "matches", max_files)
    if not files:
        print("[matches] skipping folder matches (no files found)")
        return None

    rows: list[dict[str, Any]] = []
    columns: set[str] = set()
    tracker = QualityTracker(KEY_NULL_COLUMNS["matches"], sample_rows)
    row_count = 0
    logger = ProgressPrinter(len(files), "matches", log_every_files, log_every_rows)

    for idx, path in enumerate(files, start=1):
        for record in load_json_records(path):
            row = normalize_match_row(flatten_json(record))
            rows.append(row)
            columns.update(row.keys())
            tracker.update(row)
            row_count += 1

            home_id = safe_int(row.get("home_team_id"))
            away_id = safe_int(row.get("away_team_id"))
            if home_id is not None:
                teams[home_id] = str(row.get("home_team_name") or "")
            if away_id is not None:
                teams[away_id] = str(row.get("away_team_name") or "")
        logger.maybe_log(idx, row_count)
    logger.maybe_log(len(files), row_count, force=True)

    preferred = REQUIRED_COLUMNS["matches"]
    ordered = [c for c in preferred if c in columns] + sorted(c for c in columns if c not in preferred)
    output_path = output_dir / "matches.csv"
    write_rows(output_path, ordered, [{c: row.get(c, "") for c in ordered} for row in rows])

    return {
        **finalize_table_quality("matches", row_count, set(ordered), tracker),
        "output": str(output_path),
        "files_processed": len(files),
    }

def process_lineups(
    input_dir: Path,
    output_dir: Path,
    max_files: int | None,
    log_every_files: int,
    log_every_rows: int,
    sample_rows: int | None,
    teams: dict[int, str],
    players: dict[int, tuple[str, int | None]],
) -> dict[str, Any] | None:
    files = resolve_recursive_json(input_dir, "lineups", max_files)
    if not files:
        print("[lineups] skipping folder lineups (no files found)")
        return None

    output_path = output_dir / "lineups_players.csv"
    fields = REQUIRED_COLUMNS["lineups_players"]
    tracker = QualityTracker(KEY_NULL_COLUMNS["lineups_players"], sample_rows)
    logger = ProgressPrinter(len(files), "lineups", log_every_files, log_every_rows)
    row_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for idx, path in enumerate(files, start=1):
            match_id = parse_numeric_stem(path)
            for team_record in load_json_records(path):
                team_id = safe_int(team_record.get("team_id") or team_record.get("team", {}).get("id"))
                team_name = str(team_record.get("team_name") or team_record.get("team", {}).get("name") or "")
                if team_id is not None:
                    teams[team_id] = team_name

                lineup = team_record.get("lineup")
                if not isinstance(lineup, list):
                    continue
                for player in lineup:
                    if not isinstance(player, dict):
                        continue
                    positions = player.get("positions") if isinstance(player.get("positions"), list) else []
                    first_pos = positions[0] if positions and isinstance(positions[0], dict) else {}

                    start_reason = str(first_pos.get("start_reason") or "")
                    starter = start_reason == "Starting XI"
                    player_id = safe_int(player.get("player_id"))
                    player_name = str(player.get("player_name") or "")
                    if player_id is not None:
                        players[player_id] = (player_name, team_id)

                    row = {
                        "match_id": "" if match_id is None else match_id,
                        "team_id": "" if team_id is None else team_id,
                        "team_name": team_name,
                        "player_id": "" if player_id is None else player_id,
                        "player_name": player_name,
                        "jersey_number": player.get("jersey_number", ""),
                        "position_id": first_pos.get("position_id", ""),
                        "position_name": first_pos.get("position", ""),
                        "from": first_pos.get("from", ""),
                        "to": first_pos.get("to", ""),
                        "starter": starter,
                    }
                    writer.writerow(row)
                    tracker.update(row)
                    row_count += 1
            logger.maybe_log(idx, row_count)
    logger.maybe_log(len(files), row_count, force=True)

    return {
        **finalize_table_quality("lineups_players", row_count, set(fields), tracker),
        "output": str(output_path),
        "files_processed": len(files),
    }


def get_event_uuid(row: dict[str, Any]) -> str:
    for key in ["event_uuid", "event_id", "id", "event"]:
        val = row.get(key)
        if val not in (None, ""):
            return str(val)
    return ""


def process_three_sixty(
    input_dir: Path,
    output_dir: Path,
    max_files: int | None,
    log_every_files: int,
    log_every_rows: int,
    sample_rows: int | None,
    players: dict[int, tuple[str, int | None]],
) -> dict[str, Any] | None:
    files = resolve_recursive_json(input_dir, "three-sixty", max_files)
    if not files:
        print("[three-sixty] skipping folder three-sixty (no files found)")
        return None

    schema_cols: set[str] = set()
    pass1_rows = 0
    logger1 = ProgressPrinter(len(files), "three-sixty pass 1/2", log_every_files, log_every_rows)
    for idx, path in enumerate(files, start=1):
        match_id = parse_numeric_stem(path)
        for record in load_json_records(path):
            row = flatten_json(record)
            row["match_id"] = "" if match_id is None else match_id
            row["event_uuid"] = get_event_uuid(row)
            schema_cols.update(row.keys())
            pass1_rows += 1
        logger1.maybe_log(idx, pass1_rows)
    logger1.maybe_log(len(files), pass1_rows, force=True)

    preferred = ["match_id", "event_uuid"]
    fields_main = [c for c in preferred if c in schema_cols] + sorted(c for c in schema_cols if c not in preferred)
    fields_ff = ["match_id", "event_uuid", "player_id", "teammate", "actor", "keeper", "location", "location_x", "location_y"]
    fields_vis = ["match_id", "event_uuid", "visible_area", "visible_area_point_count"]

    out_main = output_dir / "three_sixty.csv"
    out_ff = output_dir / "three_sixty_freeze_frames.csv"
    out_vis = output_dir / "three_sixty_visible_area.csv"

    tracker_main = QualityTracker(KEY_NULL_COLUMNS["three_sixty"], sample_rows)
    tracker_ff = QualityTracker(KEY_NULL_COLUMNS["three_sixty_freeze_frames"], sample_rows)
    tracker_vis = QualityTracker(KEY_NULL_COLUMNS["three_sixty_visible_area"], sample_rows)

    count_main = 0
    count_ff = 0
    count_vis = 0
    logger2 = ProgressPrinter(len(files), "three-sixty pass 2/2", log_every_files, log_every_rows)

    out_main.parent.mkdir(parents=True, exist_ok=True)
    with out_main.open("w", encoding="utf-8", newline="") as f_main, out_ff.open(
        "w", encoding="utf-8", newline=""
    ) as f_ff, out_vis.open("w", encoding="utf-8", newline="") as f_vis:
        writer_main = csv.DictWriter(f_main, fieldnames=fields_main)
        writer_ff = csv.DictWriter(f_ff, fieldnames=fields_ff)
        writer_vis = csv.DictWriter(f_vis, fieldnames=fields_vis)
        writer_main.writeheader()
        writer_ff.writeheader()
        writer_vis.writeheader()

        for idx, path in enumerate(files, start=1):
            match_id = parse_numeric_stem(path)
            for record in load_json_records(path):
                flat = flatten_json(record)
                event_uuid = get_event_uuid(flat)
                flat["match_id"] = "" if match_id is None else match_id
                flat["event_uuid"] = event_uuid
                main_row = {col: flat.get(col, "") for col in fields_main}
                writer_main.writerow(main_row)
                tracker_main.update(main_row)
                count_main += 1

                freeze = record.get("freeze_frame")
                if isinstance(freeze, list):
                    for ff in freeze:
                        if not isinstance(ff, dict):
                            continue
                        ff_flat = flatten_json(ff)
                        ff_row = {
                            "match_id": "" if match_id is None else match_id,
                            "event_uuid": event_uuid,
                            "player_id": safe_int(ff_flat.get("player_id")),
                            "teammate": ff_flat.get("teammate", ""),
                            "actor": ff_flat.get("actor", ""),
                            "keeper": ff_flat.get("keeper", ""),
                            "location": ff_flat.get("location", ""),
                            "location_x": ff_flat.get("location_x", ""),
                            "location_y": ff_flat.get("location_y", ""),
                        }
                        writer_ff.writerow(ff_row)
                        tracker_ff.update(ff_row)
                        count_ff += 1
                        pid = safe_int(ff_row.get("player_id"))
                        if pid is not None and pid not in players:
                            players[pid] = ("", None)

                visible_area = record.get("visible_area")
                if isinstance(visible_area, list):
                    vis_row = {
                        "match_id": "" if match_id is None else match_id,
                        "event_uuid": event_uuid,
                        "visible_area": json.dumps(visible_area),
                        "visible_area_point_count": int(len(visible_area) / 2),
                    }
                    writer_vis.writerow(vis_row)
                    tracker_vis.update(vis_row)
                    count_vis += 1
            logger2.maybe_log(idx, count_main)
    logger2.maybe_log(len(files), count_main, force=True)

    return {
        "main": {
            **finalize_table_quality("three_sixty", count_main, set(fields_main), tracker_main),
            "output": str(out_main),
            "files_processed": len(files),
        },
        "freeze": {
            **finalize_table_quality("three_sixty_freeze_frames", count_ff, set(fields_ff), tracker_ff),
            "output": str(out_ff),
            "files_processed": len(files),
        },
        "visible": {
            **finalize_table_quality("three_sixty_visible_area", count_vis, set(fields_vis), tracker_vis),
            "output": str(out_vis),
            "files_processed": len(files),
        },
    }

def build_dimensions(
    output_dir: Path,
    teams: dict[int, str],
    players: dict[int, tuple[str, int | None]],
    sample_rows: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    teams_path = output_dir / "teams.csv"
    players_path = output_dir / "players.csv"

    team_rows = [{"team_id": tid, "team_name": name} for tid, name in sorted(teams.items(), key=lambda x: x[0])]
    player_rows = [
        {"player_id": pid, "player_name": name, "team_id": "" if team_id is None else team_id}
        for pid, (name, team_id) in sorted(players.items(), key=lambda x: x[0])
    ]

    write_rows(teams_path, ["team_id", "team_name"], team_rows)
    write_rows(players_path, ["player_id", "player_name", "team_id"], player_rows)

    tracker_teams = QualityTracker(KEY_NULL_COLUMNS["teams"], sample_rows)
    for row in team_rows:
        tracker_teams.update(row)
    tracker_players = QualityTracker(KEY_NULL_COLUMNS["players"], sample_rows)
    for row in player_rows:
        tracker_players.update(row)

    teams_quality = {
        **finalize_table_quality("teams", len(team_rows), {"team_id", "team_name"}, tracker_teams),
        "output": str(teams_path),
    }
    players_quality = {
        **finalize_table_quality("players", len(player_rows), {"player_id", "player_name", "team_id"}, tracker_players),
        "output": str(players_path),
    }
    return teams_quality, players_quality


def write_data_quality_report(report_path: Path, tables: list[dict[str, Any]], skipped: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Quality Report", "", "## Output Row Counts"]
    for table in tables:
        lines.append(f"- `{table['table']}`: {table['row_count']:,} rows")

    if skipped:
        lines.extend(["", "## Skipped Processors"])
        for item in skipped:
            lines.append(f"- {item}")

    lines.extend(["", "## Required Column Checks"])
    for table in tables:
        missing = table["missing_required"]
        if missing:
            lines.append(f"- `{table['table']}`: FAIL (missing: {', '.join(missing)})")
        else:
            lines.append(f"- `{table['table']}`: PASS")

    lines.extend(["", "## Key ID Null Rates"])
    for table in tables:
        if not table["null_rates"]:
            continue
        parts = [f"{k}={v}" for k, v in table["null_rates"].items()]
        lines.append(f"- `{table['table']}`: " + ", ".join(parts) + f" (sampled rows: {table['sampled_rows']:,})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    include = parse_processor_list(args.include)
    exclude = parse_processor_list(args.exclude)

    output_dir.mkdir(parents=True, exist_ok=True)
    teams: dict[int, str] = {}
    players: dict[int, tuple[str, int | None]] = {}
    quality_tables: list[dict[str, Any]] = []
    skipped: list[str] = []

    if should_run_processor("events", include, exclude):
        result = process_events(
            input_dir,
            output_dir,
            args.max_files,
            args.log_every_files,
            args.log_every_rows,
            args.report_sample_rows,
            teams,
            players,
        )
        if result:
            quality_tables.append(result)
        else:
            skipped.append("events")
    else:
        skipped.append("events (excluded)")

    if should_run_processor("competitions", include, exclude):
        result = process_competitions(
            input_dir,
            output_dir,
            args.max_files,
            args.log_every_files,
            args.log_every_rows,
            args.report_sample_rows,
        )
        if result:
            quality_tables.append(result)
        else:
            skipped.append("competitions")
    else:
        skipped.append("competitions (excluded)")

    if should_run_processor("matches", include, exclude):
        result = process_matches(
            input_dir,
            output_dir,
            args.max_files,
            args.log_every_files,
            args.log_every_rows,
            args.report_sample_rows,
            teams,
        )
        if result:
            quality_tables.append(result)
        else:
            skipped.append("matches")
    else:
        skipped.append("matches (excluded)")

    if should_run_processor("lineups", include, exclude):
        result = process_lineups(
            input_dir,
            output_dir,
            args.max_files,
            args.log_every_files,
            args.log_every_rows,
            args.report_sample_rows,
            teams,
            players,
        )
        if result:
            quality_tables.append(result)
        else:
            skipped.append("lineups")
    else:
        skipped.append("lineups (excluded)")

    if should_run_processor("three-sixty", include, exclude):
        result = process_three_sixty(
            input_dir,
            output_dir,
            args.max_files,
            args.log_every_files,
            args.log_every_rows,
            args.report_sample_rows,
            players,
        )
        if result:
            quality_tables.extend([result["main"], result["freeze"], result["visible"]])
        else:
            skipped.append("three-sixty")
    else:
        skipped.append("three-sixty (excluded)")

    teams_quality, players_quality = build_dimensions(output_dir, teams, players, args.report_sample_rows)
    quality_tables.extend([teams_quality, players_quality])

    write_data_quality_report(report_path, quality_tables, skipped)

    print("ETL completed.")
    for table in quality_tables:
        print(f"- {table['table']}: {table['row_count']:,} rows -> {table.get('output', '')}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()

