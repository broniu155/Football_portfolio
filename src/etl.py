
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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

COORDINATE_KEYS = {"location", "pass_end_location", "carry_end_location", "shot_end_location"}
PROCESSORS = ["events", "competitions", "matches", "lineups", "three-sixty"]
PROCESSOR_TABLES = {
    "events": ["events"],
    "competitions": ["competitions"],
    "matches": ["matches"],
    "lineups": ["lineups_players"],
    "three-sixty": ["three_sixty", "three_sixty_freeze_frames", "three_sixty_visible_area"],
}


class ProgressPrinter:
    def __init__(self, total_files: int, phase: str, log_every_files: int, log_every_rows: int) -> None:
        self.total_files = max(total_files, 1)
        self.phase = phase
        self.log_every_files = max(1, log_every_files)
        self.log_every_rows = max(1, log_every_rows)
        self.start = time.perf_counter()
        self.last_file_log = 0
        self.last_row_log = 0

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        total = int(seconds)
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    def maybe_log(self, file_count: int, row_count: int, force: bool = False) -> None:
        should = force or (file_count - self.last_file_log >= self.log_every_files) or (
            row_count - self.last_row_log >= self.log_every_rows
        )
        if not should:
            return
        elapsed = time.perf_counter() - self.start
        rate = row_count / elapsed if elapsed > 0 else 0.0
        print(
            f"[{self.phase}] Processed files: {file_count}/{self.total_files} "
            f"| rows: {row_count:,} | rate: {rate:,.0f} rows/s | elapsed: {self._fmt_elapsed(elapsed)}"
        )
        self.last_file_log = file_count
        self.last_row_log = row_count


class QualityTracker:
    def __init__(self, key_columns: Iterable[str], sample_limit: int | None) -> None:
        self.key_columns = list(key_columns)
        self.sample_limit = sample_limit
        self.row_count = 0
        self.sampled_rows = 0
        self.null_counts = {c: 0 for c in self.key_columns}

    def update(self, row: dict[str, Any]) -> None:
        self.row_count += 1
        if self.sample_limit is not None and self.sampled_rows >= self.sample_limit:
            return
        self.sampled_rows += 1
        for c in self.key_columns:
            if row.get(c, "") in (None, ""):
                self.null_counts[c] += 1

    def null_rates(self, available: set[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for c in self.key_columns:
            if c not in available:
                out[c] = "column missing"
            elif self.sampled_rows == 0:
                out[c] = "not computed"
            else:
                out[c] = f"{(self.null_counts[c] / self.sampled_rows) * 100:.2f}%"
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StatsBomb ETL with resilient loading and incremental processing")
    p.add_argument("--input-dir", default="data_raw")
    p.add_argument("--output-dir", default="data_processed")
    p.add_argument("--report-path", default="reports/data_quality.md")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--log-every-files", type=int, default=50)
    p.add_argument("--log-every-rows", type=int, default=100000)
    p.add_argument("--report-sample-rows", type=int, default=None)
    p.add_argument("--include", default=None)
    p.add_argument("--exclude", default=None)
    p.add_argument("--strict-json", action="store_true", help="Fail fast on malformed JSON")
    p.add_argument("--quarantine-dir", default="reports/quarantine")
    p.add_argument("--state-path", default=None, help="Defaults to <output-dir>/.etl_state.json")
    p.add_argument("--incremental", dest="incremental", action="store_true")
    p.add_argument("--no-incremental", dest="incremental", action="store_false")
    p.set_defaults(incremental=True)
    p.add_argument("--force", action="store_true", help="Ignore state and process all files")
    p.add_argument("--append", dest="append", action="store_true")
    p.add_argument("--no-append", dest="append", action="store_false")
    p.set_defaults(append=True)
    p.add_argument("--state-hash", action="store_true", help="Include SHA1 in incremental fingerprint")
    return p.parse_args()


def parse_processor_list(raw: str | None) -> set[str]:
    if raw is None or raw.strip() == "":
        return set()
    vals = {x.strip() for x in raw.split(",") if x.strip()}
    invalid = sorted(vals - set(PROCESSORS))
    if invalid:
        raise ValueError(f"Unknown processor(s): {invalid}. Valid: {PROCESSORS}")
    return vals


def should_run(name: str, include: set[str], exclude: set[str]) -> bool:
    if include and name not in include:
        return False
    return name not in exclude


def maybe_limit(files: list[Path], max_files: int | None) -> list[Path]:
    return files[:max_files] if max_files is not None and max_files > 0 else files


def resolve_events_files(input_dir: Path, max_files: int | None) -> list[Path]:
    events_dir = input_dir / "events"
    files = sorted(events_dir.glob("*.json")) if events_dir.exists() else sorted(input_dir.glob("*.json"))
    return maybe_limit(files, max_files)


def resolve_competitions_files(input_dir: Path, max_files: int | None) -> list[Path]:
    files: list[Path] = []
    cjson = input_dir / "competitions.json"
    if cjson.exists():
        files.append(cjson)
    cdir = input_dir / "competitions"
    if cdir.exists():
        files.extend(sorted(cdir.rglob("*.json")))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return maybe_limit(uniq, max_files)


def resolve_recursive_json(input_dir: Path, folder: str, max_files: int | None) -> list[Path]:
    target = input_dir / folder
    files = sorted(target.rglob("*.json")) if target.exists() else []
    return maybe_limit(files, max_files)


def rel_key(path: Path, input_root: Path) -> str:
    try:
        return path.relative_to(input_root).as_posix()
    except ValueError:
        return path.as_posix()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "files": {}}
        data.setdefault("version", 1)
        data.setdefault("files", {})
        return data
    except Exception:
        return {"version": 1, "files": {}}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def file_fingerprint(path: Path, use_hash: bool) -> dict[str, Any]:
    st = path.stat()
    fp = {"size_bytes": int(st.st_size), "mtime": int(st.st_mtime)}
    if use_hash:
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        fp["sha1"] = h.hexdigest()
    return fp


def filter_incremental_files(
    processor: str,
    files: list[Path],
    input_root: Path,
    state: dict[str, Any],
    incremental: bool,
    force: bool,
    state_hash: bool,
) -> tuple[list[Path], dict[str, dict[str, Any]], int]:
    proc_state = state.get("files", {}).get(processor, {}) if isinstance(state.get("files"), dict) else {}
    to_process: list[Path] = []
    fingerprints: dict[str, dict[str, Any]] = {}
    skipped_unchanged = 0
    for path in files:
        rel = rel_key(path, input_root)
        fp = file_fingerprint(path, state_hash)
        fingerprints[rel] = fp
        if incremental and not force and proc_state.get(rel) == fp:
            skipped_unchanged += 1
            continue
        to_process.append(path)
    return to_process, fingerprints, skipped_unchanged


def parse_numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def is_numeric_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, (int, float)) for v in value)


def flatten_json(source: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in source.items():
        out_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(flatten_json(value, out_key))
            continue
        if is_numeric_list(value) and len(value) in (2, 3):
            is_coord = out_key in COORDINATE_KEYS or out_key.endswith("_location")
            flat[out_key] = json.dumps(value)
            if is_coord:
                axes = ["x", "y", "z"]
                for idx, item in enumerate(value):
                    flat[f"{out_key}_{axes[idx] if idx < 3 else f'coord_{idx + 1}'}"] = item
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


def existing_header(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader, None)


def open_csv_writer(path: Path, fieldnames: list[str], append: bool) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    use_append = append and path.exists()
    mode = "a" if use_append else "w"
    f = path.open(mode, encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not use_append:
        w.writeheader()
    return f, w


def load_existing_dimensions(
    output_dir: Path,
    teams: dict[int, str],
    players: dict[int, tuple[str, int | None]],
) -> None:
    teams_path = output_dir / "teams.csv"
    players_path = output_dir / "players.csv"

    if teams_path.exists():
        with teams_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                team_id = safe_int(row.get("team_id"))
                if team_id is not None:
                    teams[team_id] = str(row.get("team_name") or "")

    if players_path.exists():
        with players_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                player_id = safe_int(row.get("player_id"))
                if player_id is None:
                    continue
                team_id = safe_int(row.get("team_id"))
                players[player_id] = (str(row.get("player_name") or ""), team_id)


def update_team_player_from_row(row: dict[str, Any], teams: dict[int, str], players: dict[int, tuple[str, int | None]]) -> None:
    team_id = safe_int(row.get("team_id"))
    if team_id is not None:
        teams[team_id] = str(row.get("team_name") or "")
    player_id = safe_int(row.get("player_id"))
    if player_id is not None:
        players[player_id] = (str(row.get("player_name") or ""), team_id)


def finalize_table_quality(table: str, rows_written: int, available_columns: set[str], tracker: QualityTracker) -> dict[str, Any]:
    missing = [c for c in REQUIRED_COLUMNS.get(table, []) if c not in available_columns]
    return {
        "table": table,
        "row_count": rows_written,
        "missing_required": missing,
        "null_rates": tracker.null_rates(available_columns),
        "sampled_rows": tracker.sampled_rows,
    }


def quarantine_file(path: Path, quarantine_dir: Path, processor: str) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    safe_name = path.name.replace(":", "_")
    dst = quarantine_dir / processor / safe_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(path, dst)
    except Exception:
        pass


def load_json_records(
    path: Path,
    *,
    strict_json: bool,
    quarantine_dir: Path,
    processor: str,
    parse_failure_counter: dict[str, int],
    failed_files: set[str] | None = None,
) -> list[dict[str, Any]]:
    try:
        if path.stat().st_size == 0:
            raise ValueError("empty file")
        with path.open("r", encoding="utf-8-sig") as f:
            obj = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, IOError, ValueError) as exc:
        parse_failure_counter[processor] = parse_failure_counter.get(processor, 0) + 1
        if failed_files is not None:
            failed_files.add(str(path))
        print(f"[WARN] Failed to parse JSON: {path} | {exc} | skipping file")
        quarantine_file(path, quarantine_dir, processor)
        if strict_json:
            raise
        return []

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []

def process_events(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    parse_failures: dict[str, int],
    teams: dict[int, str],
    players: dict[int, tuple[str, int | None]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    proc = "events"
    all_files = resolve_events_files(input_dir, args.max_files)
    to_process, fingerprints, skipped_unchanged = filter_incremental_files(
        proc, all_files, input_dir, state, args.incremental, args.force, args.state_hash
    )
    print(f"[events] Found: {len(all_files)} files | Skipping unchanged: {skipped_unchanged} | Processing: {len(to_process)}")
    summary = {"found": len(all_files), "skipped_unchanged": skipped_unchanged, "processed": len(to_process)}
    if not to_process:
        return [], summary, {}
    failed_files: set[str] = set()
    failed_files: set[str] = set()

    # pass 1: schema + dims
    cols: set[str] = set()
    logger1 = ProgressPrinter(len(to_process), "events pass 1/2", args.log_every_files, args.log_every_rows)
    row_count = 0
    for i, path in enumerate(to_process, start=1):
        mid = parse_numeric_stem(path)
        if mid is None:
            raise ValueError(f"Cannot parse match_id from filename: {path.name}")
        for rec in load_json_records(path, strict_json=args.strict_json, quarantine_dir=Path(args.quarantine_dir), processor=proc, parse_failure_counter=parse_failures, failed_files=failed_files):
            row = flatten_json(rec)
            row["match_id"] = mid
            if "id" in row and "event_id" not in row:
                row["event_id"] = row.pop("id")
            if "index" in row and "event_index" not in row:
                row["event_index"] = row.pop("index")
            for c in ["match_id", "team_id", "player_id", "type_id", "minute", "second"]:
                row[c] = safe_int(row.get(c))
            cols.update(row.keys())
            update_team_player_from_row(row, teams, players)
            row_count += 1
        logger1.maybe_log(i, row_count)
    logger1.maybe_log(len(to_process), row_count, force=True)

    missing = [c for c in REQUIRED_COLUMNS[proc] if c not in cols]
    if missing:
        raise ValueError(f"[events] missing required columns after flattening: {missing}")

    preferred = [
        "event_id", "match_id", "event_index", "period", "timestamp", "minute", "second", "team_id", "team_name",
        "player_id", "player_name", "type_id", "type_name", "location", "location_x", "location_y", "location_z",
        "pass_end_location", "pass_end_location_x", "pass_end_location_y", "pass_end_location_z",
    ]
    header = existing_header(output_dir / "events.csv") if args.append else None
    ordered = header if header else ([c for c in preferred if c in cols] + sorted(c for c in cols if c not in preferred))

    tracker = QualityTracker(KEY_NULL_COLUMNS[proc], args.report_sample_rows)
    rows_written = 0
    logger2 = ProgressPrinter(len(to_process), "events pass 2/2", args.log_every_files, args.log_every_rows)
    f, w = open_csv_writer(output_dir / "events.csv", ordered, args.append)
    try:
        for i, path in enumerate(to_process, start=1):
            mid = int(path.stem)
            for rec in load_json_records(path, strict_json=args.strict_json, quarantine_dir=Path(args.quarantine_dir), processor=proc, parse_failure_counter=parse_failures, failed_files=failed_files):
                row = flatten_json(rec)
                row["match_id"] = mid
                if "id" in row and "event_id" not in row:
                    row["event_id"] = row.pop("id")
                if "index" in row and "event_index" not in row:
                    row["event_index"] = row.pop("index")
                for c in ["match_id", "team_id", "player_id", "type_id", "minute", "second"]:
                    row[c] = safe_int(row.get(c))
                out = {c: row.get(c, "") for c in ordered}
                w.writerow(out)
                tracker.update(out)
                rows_written += 1
            logger2.maybe_log(i, rows_written)
    finally:
        f.close()
    logger2.maybe_log(len(to_process), rows_written, force=True)

    tables = [finalize_table_quality(proc, rows_written, set(ordered), tracker)]
    tables[0]["output"] = str(output_dir / "events.csv")
    return tables, summary, {
        rel_key(p, input_dir): fingerprints[rel_key(p, input_dir)]
        for p in to_process
        if str(p) not in failed_files
    }


def _scan_table_schema(
    files: list[Path],
    processor: str,
    args: argparse.Namespace,
    parse_failures: dict[str, int],
    row_fn,
) -> set[str]:
    cols: set[str] = set()
    logger = ProgressPrinter(len(files), f"{processor} schema", args.log_every_files, args.log_every_rows)
    rows = 0
    for i, path in enumerate(files, start=1):
        for rec in load_json_records(path, strict_json=args.strict_json, quarantine_dir=Path(args.quarantine_dir), processor=processor, parse_failure_counter=parse_failures):
            row = row_fn(path, rec)
            cols.update(row.keys())
            rows += 1
        logger.maybe_log(i, rows)
    logger.maybe_log(len(files), rows, force=True)
    return cols


def process_competitions(input_dir: Path, output_dir: Path, args: argparse.Namespace, state: dict[str, Any], parse_failures: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    proc = "competitions"
    all_files = resolve_competitions_files(input_dir, args.max_files)
    to_process, fingerprints, skipped_unchanged = filter_incremental_files(proc, all_files, input_dir, state, args.incremental, args.force, args.state_hash)
    print(f"[competitions] Found: {len(all_files)} files | Skipping unchanged: {skipped_unchanged} | Processing: {len(to_process)}")
    summary = {"found": len(all_files), "skipped_unchanged": skipped_unchanged, "processed": len(to_process)}
    if not to_process:
        return [], summary, {}
    failed_files: set[str] = set()

    def mk_row(_: Path, rec: dict[str, Any]) -> dict[str, Any]:
        row = flatten_json(rec)
        row.setdefault("season_id", row.get("season_season_id"))
        row.setdefault("season_name", row.get("season_season_name"))
        row.setdefault("competition_id", row.get("competition_competition_id"))
        row.setdefault("competition_name", row.get("competition_competition_name"))
        row.setdefault("country_name", row.get("competition_country_name") or row.get("country_name"))
        return row

    header = existing_header(output_dir / "competitions.csv") if args.append else None
    if header:
        ordered = header
    else:
        cols = _scan_table_schema(to_process, proc, args, parse_failures, mk_row)
        pref = REQUIRED_COLUMNS[proc]
        ordered = [c for c in pref if c in cols] + sorted(c for c in cols if c not in pref)

    tracker = QualityTracker(KEY_NULL_COLUMNS[proc], args.report_sample_rows)
    rows_written = 0
    logger = ProgressPrinter(len(to_process), proc, args.log_every_files, args.log_every_rows)
    f, w = open_csv_writer(output_dir / "competitions.csv", ordered, args.append)
    try:
        for i, path in enumerate(to_process, start=1):
            for rec in load_json_records(path, strict_json=args.strict_json, quarantine_dir=Path(args.quarantine_dir), processor=proc, parse_failure_counter=parse_failures, failed_files=failed_files):
                out = {c: mk_row(path, rec).get(c, "") for c in ordered}
                w.writerow(out)
                tracker.update(out)
                rows_written += 1
            logger.maybe_log(i, rows_written)
    finally:
        f.close()
    logger.maybe_log(len(to_process), rows_written, force=True)

    table = finalize_table_quality(proc, rows_written, set(ordered), tracker)
    table["output"] = str(output_dir / "competitions.csv")
    return [table], summary, {
        rel_key(p, input_dir): fingerprints[rel_key(p, input_dir)]
        for p in to_process
        if str(p) not in failed_files
    }


def process_matches(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    parse_failures: dict[str, int],
    teams: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    proc = "matches"
    all_files = resolve_recursive_json(input_dir, "matches", args.max_files)
    to_process, fingerprints, skipped_unchanged = filter_incremental_files(proc, all_files, input_dir, state, args.incremental, args.force, args.state_hash)
    print(f"[matches] Found: {len(all_files)} files | Skipping unchanged: {skipped_unchanged} | Processing: {len(to_process)}")
    summary = {"found": len(all_files), "skipped_unchanged": skipped_unchanged, "processed": len(to_process)}
    if not to_process:
        return [], summary, {}
    failed_files: set[str] = set()

    def mk_row(_: Path, rec: dict[str, Any]) -> dict[str, Any]:
        row = flatten_json(rec)
        row["match_id"] = safe_int(row.get("match_id") or row.get("match_match_id"))
        row["competition_id"] = safe_int(row.get("competition_id") or row.get("competition_competition_id"))
        row["season_id"] = safe_int(row.get("season_id") or row.get("season_season_id"))
        row["home_team_id"] = safe_int(row.get("home_team_id") or row.get("home_team_home_team_id"))
        row["home_team_name"] = row.get("home_team_name") or row.get("home_team_home_team_name") or ""
        row["away_team_id"] = safe_int(row.get("away_team_id") or row.get("away_team_away_team_id"))
        row["away_team_name"] = row.get("away_team_name") or row.get("away_team_away_team_name") or ""
        return row

    header = existing_header(output_dir / "matches.csv") if args.append else None
    if header:
        ordered = header
    else:
        cols = _scan_table_schema(to_process, proc, args, parse_failures, mk_row)
        pref = REQUIRED_COLUMNS[proc]
        ordered = [c for c in pref if c in cols] + sorted(c for c in cols if c not in pref)

    tracker = QualityTracker(KEY_NULL_COLUMNS[proc], args.report_sample_rows)
    rows_written = 0
    logger = ProgressPrinter(len(to_process), proc, args.log_every_files, args.log_every_rows)
    f, w = open_csv_writer(output_dir / "matches.csv", ordered, args.append)
    try:
        for i, path in enumerate(to_process, start=1):
            for rec in load_json_records(path, strict_json=args.strict_json, quarantine_dir=Path(args.quarantine_dir), processor=proc, parse_failure_counter=parse_failures, failed_files=failed_files):
                row = mk_row(path, rec)
                hid, aid = safe_int(row.get("home_team_id")), safe_int(row.get("away_team_id"))
                if hid is not None:
                    teams[hid] = str(row.get("home_team_name") or "")
                if aid is not None:
                    teams[aid] = str(row.get("away_team_name") or "")
                out = {c: row.get(c, "") for c in ordered}
                w.writerow(out)
                tracker.update(out)
                rows_written += 1
            logger.maybe_log(i, rows_written)
    finally:
        f.close()
    logger.maybe_log(len(to_process), rows_written, force=True)

    table = finalize_table_quality(proc, rows_written, set(ordered), tracker)
    table["output"] = str(output_dir / "matches.csv")
    return [table], summary, {
        rel_key(p, input_dir): fingerprints[rel_key(p, input_dir)]
        for p in to_process
        if str(p) not in failed_files
    }

def process_lineups(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    parse_failures: dict[str, int],
    teams: dict[int, str],
    players: dict[int, tuple[str, int | None]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    proc = "lineups"
    table_name = "lineups_players"
    all_files = resolve_recursive_json(input_dir, "lineups", args.max_files)
    to_process, fingerprints, skipped_unchanged = filter_incremental_files(proc, all_files, input_dir, state, args.incremental, args.force, args.state_hash)
    print(f"[lineups] Found: {len(all_files)} files | Skipping unchanged: {skipped_unchanged} | Processing: {len(to_process)}")
    summary = {"found": len(all_files), "skipped_unchanged": skipped_unchanged, "processed": len(to_process)}
    if not to_process:
        return [], summary, {}
    failed_files: set[str] = set()

    fields = REQUIRED_COLUMNS[table_name]
    header = existing_header(output_dir / "lineups_players.csv") if args.append else None
    ordered = header if header else fields

    tracker = QualityTracker(KEY_NULL_COLUMNS[table_name], args.report_sample_rows)
    rows_written = 0
    logger = ProgressPrinter(len(to_process), proc, args.log_every_files, args.log_every_rows)
    f, w = open_csv_writer(output_dir / "lineups_players.csv", ordered, args.append)
    try:
        for i, path in enumerate(to_process, start=1):
            mid = parse_numeric_stem(path)
            for team_record in load_json_records(path, strict_json=args.strict_json, quarantine_dir=Path(args.quarantine_dir), processor=proc, parse_failure_counter=parse_failures, failed_files=failed_files):
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
                    pos = player.get("positions") if isinstance(player.get("positions"), list) else []
                    first = pos[0] if pos and isinstance(pos[0], dict) else {}
                    player_id = safe_int(player.get("player_id"))
                    player_name = str(player.get("player_name") or "")
                    if player_id is not None:
                        players[player_id] = (player_name, team_id)
                    row = {
                        "match_id": "" if mid is None else mid,
                        "team_id": "" if team_id is None else team_id,
                        "team_name": team_name,
                        "player_id": "" if player_id is None else player_id,
                        "player_name": player_name,
                        "jersey_number": player.get("jersey_number", ""),
                        "position_id": first.get("position_id", ""),
                        "position_name": first.get("position", ""),
                        "from": first.get("from", ""),
                        "to": first.get("to", ""),
                        "starter": str(first.get("start_reason") or "") == "Starting XI",
                    }
                    out = {c: row.get(c, "") for c in ordered}
                    w.writerow(out)
                    tracker.update(out)
                    rows_written += 1
            logger.maybe_log(i, rows_written)
    finally:
        f.close()
    logger.maybe_log(len(to_process), rows_written, force=True)

    table = finalize_table_quality(table_name, rows_written, set(ordered), tracker)
    table["output"] = str(output_dir / "lineups_players.csv")
    return [table], summary, {
        rel_key(p, input_dir): fingerprints[rel_key(p, input_dir)]
        for p in to_process
        if str(p) not in failed_files
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
    args: argparse.Namespace,
    state: dict[str, Any],
    parse_failures: dict[str, int],
    players: dict[int, tuple[str, int | None]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    proc = "three-sixty"
    all_files = resolve_recursive_json(input_dir, "three-sixty", args.max_files)
    to_process, fingerprints, skipped_unchanged = filter_incremental_files(proc, all_files, input_dir, state, args.incremental, args.force, args.state_hash)
    print(f"[three-sixty] Found: {len(all_files)} files | Skipping unchanged: {skipped_unchanged} | Processing: {len(to_process)}")
    summary = {"found": len(all_files), "skipped_unchanged": skipped_unchanged, "processed": len(to_process)}
    if not to_process:
        return [], summary, {}

    # schema pass only for changed files
    cols: set[str] = set()
    logger1 = ProgressPrinter(len(to_process), "three-sixty schema", args.log_every_files, args.log_every_rows)
    rows = 0
    for i, path in enumerate(to_process, start=1):
        mid = parse_numeric_stem(path)
        for rec in load_json_records(path, strict_json=args.strict_json, quarantine_dir=Path(args.quarantine_dir), processor=proc, parse_failure_counter=parse_failures, failed_files=failed_files):
            row = flatten_json(rec)
            row["match_id"] = "" if mid is None else mid
            row["event_uuid"] = get_event_uuid(row)
            cols.update(row.keys())
            rows += 1
        logger1.maybe_log(i, rows)
    logger1.maybe_log(len(to_process), rows, force=True)

    header_main = existing_header(output_dir / "three_sixty.csv") if args.append else None
    fields_main = header_main if header_main else ([c for c in ["match_id", "event_uuid"] if c in cols] + sorted(c for c in cols if c not in {"match_id", "event_uuid"}))
    fields_ff = existing_header(output_dir / "three_sixty_freeze_frames.csv") if args.append else None
    if not fields_ff:
        fields_ff = ["match_id", "event_uuid", "player_id", "teammate", "actor", "keeper", "location", "location_x", "location_y"]
    fields_vis = existing_header(output_dir / "three_sixty_visible_area.csv") if args.append else None
    if not fields_vis:
        fields_vis = ["match_id", "event_uuid", "visible_area", "visible_area_point_count"]

    t_main = QualityTracker(KEY_NULL_COLUMNS["three_sixty"], args.report_sample_rows)
    t_ff = QualityTracker(KEY_NULL_COLUMNS["three_sixty_freeze_frames"], args.report_sample_rows)
    t_vis = QualityTracker(KEY_NULL_COLUMNS["three_sixty_visible_area"], args.report_sample_rows)

    c_main = c_ff = c_vis = 0
    logger2 = ProgressPrinter(len(to_process), "three-sixty write", args.log_every_files, args.log_every_rows)
    f_main, w_main = open_csv_writer(output_dir / "three_sixty.csv", fields_main, args.append)
    f_ff, w_ff = open_csv_writer(output_dir / "three_sixty_freeze_frames.csv", fields_ff, args.append)
    f_vis, w_vis = open_csv_writer(output_dir / "three_sixty_visible_area.csv", fields_vis, args.append)
    try:
        for i, path in enumerate(to_process, start=1):
            mid = parse_numeric_stem(path)
            for rec in load_json_records(path, strict_json=args.strict_json, quarantine_dir=Path(args.quarantine_dir), processor=proc, parse_failure_counter=parse_failures, failed_files=failed_files):
                flat = flatten_json(rec)
                event_uuid = get_event_uuid(flat)
                flat["match_id"] = "" if mid is None else mid
                flat["event_uuid"] = event_uuid

                row_main = {c: flat.get(c, "") for c in fields_main}
                w_main.writerow(row_main)
                t_main.update(row_main)
                c_main += 1

                ff_list = rec.get("freeze_frame")
                if isinstance(ff_list, list):
                    for ff in ff_list:
                        if not isinstance(ff, dict):
                            continue
                        fff = flatten_json(ff)
                        row_ff = {
                            "match_id": "" if mid is None else mid,
                            "event_uuid": event_uuid,
                            "player_id": safe_int(fff.get("player_id")),
                            "teammate": fff.get("teammate", ""),
                            "actor": fff.get("actor", ""),
                            "keeper": fff.get("keeper", ""),
                            "location": fff.get("location", ""),
                            "location_x": fff.get("location_x", ""),
                            "location_y": fff.get("location_y", ""),
                        }
                        w_ff.writerow({c: row_ff.get(c, "") for c in fields_ff})
                        t_ff.update(row_ff)
                        c_ff += 1
                        pid = safe_int(row_ff.get("player_id"))
                        if pid is not None and pid not in players:
                            players[pid] = ("", None)

                vis = rec.get("visible_area")
                if isinstance(vis, list):
                    row_vis = {
                        "match_id": "" if mid is None else mid,
                        "event_uuid": event_uuid,
                        "visible_area": json.dumps(vis),
                        "visible_area_point_count": int(len(vis) / 2),
                    }
                    w_vis.writerow({c: row_vis.get(c, "") for c in fields_vis})
                    t_vis.update(row_vis)
                    c_vis += 1
            logger2.maybe_log(i, c_main)
    finally:
        f_main.close(); f_ff.close(); f_vis.close()
    logger2.maybe_log(len(to_process), c_main, force=True)

    tables = [
        finalize_table_quality("three_sixty", c_main, set(fields_main), t_main),
        finalize_table_quality("three_sixty_freeze_frames", c_ff, set(fields_ff), t_ff),
        finalize_table_quality("three_sixty_visible_area", c_vis, set(fields_vis), t_vis),
    ]
    tables[0]["output"] = str(output_dir / "three_sixty.csv")
    tables[1]["output"] = str(output_dir / "three_sixty_freeze_frames.csv")
    tables[2]["output"] = str(output_dir / "three_sixty_visible_area.csv")
    return tables, summary, {
        rel_key(p, input_dir): fingerprints[rel_key(p, input_dir)]
        for p in to_process
        if str(p) not in failed_files
    }

def build_dimensions(
    output_dir: Path,
    append: bool,
    sample_rows: int | None,
    teams: dict[int, str],
    players: dict[int, tuple[str, int | None]],
) -> list[dict[str, Any]]:
    team_rows = [{"team_id": tid, "team_name": name} for tid, name in sorted(teams.items(), key=lambda x: x[0])]
    player_rows = [
        {"player_id": pid, "player_name": name, "team_id": "" if team_id is None else team_id}
        for pid, (name, team_id) in sorted(players.items(), key=lambda x: x[0])
    ]

    # rewrite dimensions each run to keep deduped/latest view
    write_mode_append = False if append else False
    f_t, w_t = open_csv_writer(output_dir / "teams.csv", ["team_id", "team_name"], write_mode_append)
    f_p, w_p = open_csv_writer(output_dir / "players.csv", ["player_id", "player_name", "team_id"], write_mode_append)
    try:
        for r in team_rows:
            w_t.writerow(r)
        for r in player_rows:
            w_p.writerow(r)
    finally:
        f_t.close(); f_p.close()

    qt = QualityTracker(KEY_NULL_COLUMNS["teams"], sample_rows)
    qp = QualityTracker(KEY_NULL_COLUMNS["players"], sample_rows)
    for r in team_rows:
        qt.update(r)
    for r in player_rows:
        qp.update(r)

    t = finalize_table_quality("teams", len(team_rows), {"team_id", "team_name"}, qt)
    p = finalize_table_quality("players", len(player_rows), {"player_id", "player_name", "team_id"}, qp)
    t["output"] = str(output_dir / "teams.csv")
    p["output"] = str(output_dir / "players.csv")
    return [t, p]


def write_data_quality_report(report_path: Path, tables: list[dict[str, Any]], processor_summaries: dict[str, dict[str, int]], parse_failures: dict[str, int], append_mode: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Quality Report", "", "## Run Mode", f"- Append mode: {append_mode}", ""]

    lines.append("## Processor File Summary")
    for proc in PROCESSORS:
        s = processor_summaries.get(proc, {"found": 0, "skipped_unchanged": 0, "processed": 0})
        lines.append(
            f"- `{proc}`: found={s['found']}, skipped_unchanged={s['skipped_unchanged']}, processed={s['processed']}, parse_failures={parse_failures.get(proc, 0)}"
        )

    lines.extend(["", "## Output Row Counts (This Run)"])
    for t in tables:
        lines.append(f"- `{t['table']}`: {t['row_count']:,} rows")

    lines.extend(["", "## Required Column Checks"])
    for t in tables:
        if t["missing_required"]:
            lines.append(f"- `{t['table']}`: FAIL (missing: {', '.join(t['missing_required'])})")
        else:
            lines.append(f"- `{t['table']}`: PASS")

    lines.extend(["", "## Key ID Null Rates"])
    for t in tables:
        parts = [f"{k}={v}" for k, v in t["null_rates"].items()]
        if parts:
            lines.append(f"- `{t['table']}`: " + ", ".join(parts) + f" (sampled rows: {t['sampled_rows']:,})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    state_path = Path(args.state_path) if args.state_path else output_dir / ".etl_state.json"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    include = parse_processor_list(args.include)
    exclude = parse_processor_list(args.exclude)

    if not args.append and args.incremental and not args.force:
        print("[INFO] --no-append with incremental can produce partial rewrites. Enabling --force for this run.")
        args.force = True

    state = load_state(state_path)
    state.setdefault("files", {})

    parse_failures: dict[str, int] = {p: 0 for p in PROCESSORS}
    processor_summaries: dict[str, dict[str, int]] = {}
    quality_tables: list[dict[str, Any]] = []
    teams: dict[int, str] = {}
    players: dict[int, tuple[str, int | None]] = {}

    if args.append:
        load_existing_dimensions(output_dir, teams, players)

    def run_processor(name: str, fn):
        if not should_run(name, include, exclude):
            processor_summaries[name] = {"found": 0, "skipped_unchanged": 0, "processed": 0}
            for t in PROCESSOR_TABLES.get(name, []):
                quality_tables.append(
                    {
                        "table": t,
                        "row_count": 0,
                        "missing_required": [],
                        "null_rates": {},
                        "sampled_rows": 0,
                        "output": str(output_dir / f"{t}.csv"),
                    }
                )
            return
        tables, summary, updates = fn()
        processor_summaries[name] = summary
        if tables:
            quality_tables.extend(tables)
        else:
            for t in PROCESSOR_TABLES.get(name, []):
                quality_tables.append(
                    {
                        "table": t,
                        "row_count": 0,
                        "missing_required": [],
                        "null_rates": {},
                        "sampled_rows": 0,
                        "output": str(output_dir / f"{t}.csv"),
                    }
                )
        if updates:
            state["files"].setdefault(name, {}).update(updates)
            save_state(state_path, state)

    run_processor(
        "events",
        lambda: process_events(input_dir, output_dir, args, state, parse_failures, teams, players),
    )
    run_processor(
        "competitions",
        lambda: process_competitions(input_dir, output_dir, args, state, parse_failures),
    )
    run_processor(
        "matches",
        lambda: process_matches(input_dir, output_dir, args, state, parse_failures, teams),
    )
    run_processor(
        "lineups",
        lambda: process_lineups(input_dir, output_dir, args, state, parse_failures, teams, players),
    )
    run_processor(
        "three-sixty",
        lambda: process_three_sixty(input_dir, output_dir, args, state, parse_failures, players),
    )

    quality_tables.extend(build_dimensions(output_dir, args.append, args.report_sample_rows, teams, players))
    write_data_quality_report(report_path, quality_tables, processor_summaries, parse_failures, args.append)

    print("ETL completed.")
    for t in quality_tables:
        print(f"- {t['table']}: {t['row_count']:,} rows -> {t.get('output', '')}")
    print(f"Wrote report: {report_path}")
    print(f"State file: {state_path}")


if __name__ == "__main__":
    main()
