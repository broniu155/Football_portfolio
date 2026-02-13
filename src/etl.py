from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "type_id",
    "type_name",
    "minute",
    "second",
]

COORDINATE_KEYS = {
    "location",
    "pass_end_location",
    "carry_end_location",
    "shot_end_location",
}

REPORT_NULL_COLUMNS = [
    "match_id",
    "team_id",
    "player_id",
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
]


class ProgressPrinter:
    def __init__(
        self,
        total_files: int,
        phase: str,
        log_every_files: int,
        log_every_rows: int,
    ) -> None:
        self.total_files = total_files
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
            f"| event rows: {row_count:,} "
            f"| rate: {rate:,.0f} rows/s "
            f"| elapsed: {self._format_elapsed(elapsed)}"
        )
        self.last_file_log = file_count
        self.last_row_log = row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load StatsBomb event JSON files, flatten nested fields, "
            "write clean CSV outputs, and produce a data quality report."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data_raw",
        help="Directory containing StatsBomb JSON files (or an events subfolder).",
    )
    parser.add_argument(
        "--output-dir",
        default="data_processed",
        help="Directory where cleaned CSV tables will be written.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/data_quality.md",
        help="Path for the markdown data quality report.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for number of event files to process (useful for quick runs).",
    )
    parser.add_argument(
        "--log-every-files",
        type=int,
        default=50,
        help="Progress log interval in files.",
    )
    parser.add_argument(
        "--log-every-rows",
        type=int,
        default=100000,
        help="Progress log interval in event rows.",
    )
    parser.add_argument(
        "--report-sample-rows",
        type=int,
        default=None,
        help=(
            "Optional sample row limit for report stats. "
            "If omitted, report stats are computed on all written rows."
        ),
    )
    return parser.parse_args()


def resolve_event_files(input_dir: Path, max_files: int | None = None) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    events_dir = input_dir / "events"
    files = sorted(events_dir.glob("*.json")) if events_dir.exists() else sorted(input_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No event JSON files found in {input_dir} or {events_dir}."
        )

    if max_files is not None and max_files > 0:
        files = files[:max_files]

    return files


def is_numeric_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, (int, float)) for v in value)


def flatten_event(source: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}

    for key, value in source.items():
        out_key = f"{parent_key}_{key}" if parent_key else key

        if isinstance(value, dict):
            flat.update(flatten_event(value, out_key))
            continue

        if is_numeric_list(value) and out_key in COORDINATE_KEYS:
            flat[out_key] = json.dumps(value)
            axis = ["x", "y", "z"]
            for idx, item in enumerate(value):
                suffix = axis[idx] if idx < len(axis) else f"coord_{idx + 1}"
                flat[f"{out_key}_{suffix}"] = item
            continue

        if isinstance(value, list):
            flat[out_key] = json.dumps(value)
            continue

        flat[out_key] = value

    return flat


def normalize_event(flat: dict[str, Any], match_id: int) -> dict[str, Any]:
    row = dict(flat)

    if "id" in row and "event_id" not in row:
        row["event_id"] = row.pop("id")
    if "index" in row and "event_index" not in row:
        row["event_index"] = row.pop("index")

    row["match_id"] = match_id
    return row


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def process_record(record: dict[str, Any], match_id: int) -> dict[str, Any]:
    row = normalize_event(flatten_event(record), match_id)
    for col in ["match_id", "team_id", "player_id", "type_id", "minute", "second"]:
        row[col] = safe_int(row.get(col))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_null(value: Any) -> bool:
    return value in (None, "")


def scan_schema_and_dimensions(
    event_files: list[Path],
    log_every_files: int,
    log_every_rows: int,
) -> tuple[set[str], dict[int, str], dict[int, tuple[str | None, int | None]], int, set[int]]:
    all_columns: set[str] = set()
    teams: dict[int, str] = {}
    players: dict[int, tuple[str | None, int | None]] = {}
    total_rows = 0
    unique_matches: set[int] = set()

    logger = ProgressPrinter(
        total_files=len(event_files),
        phase="Pass 1/2",
        log_every_files=log_every_files,
        log_every_rows=log_every_rows,
    )

    for idx, event_file in enumerate(event_files, start=1):
        try:
            match_id = int(event_file.stem)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse match_id from filename: {event_file.name}. "
                "Expected numeric filename like 12345.json"
            ) from exc

        unique_matches.add(match_id)

        with event_file.open("r", encoding="utf-8-sig") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected list of events in {event_file}")

        for record in records:
            row = process_record(record, match_id)
            all_columns.update(row.keys())
            total_rows += 1

            team_id = row.get("team_id")
            if team_id is not None:
                teams[team_id] = row.get("team_name") or ""

            player_id = row.get("player_id")
            if player_id is not None:
                players[player_id] = (row.get("player_name"), team_id)

        logger.maybe_log(file_count=idx, row_count=total_rows)

    logger.maybe_log(file_count=len(event_files), row_count=total_rows, force=True)
    return all_columns, teams, players, total_rows, unique_matches


def stream_write_events_csv(
    event_files: list[Path],
    output_path: Path,
    ordered_columns: list[str],
    log_every_files: int,
    log_every_rows: int,
    report_sample_rows: int | None = None,
) -> tuple[int, dict[str, int]]:
    total_rows = 0
    null_counts = {col: 0 for col in REPORT_NULL_COLUMNS}

    logger = ProgressPrinter(
        total_files=len(event_files),
        phase="Pass 2/2",
        log_every_files=log_every_files,
        log_every_rows=log_every_rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=ordered_columns)
        writer.writeheader()

        for idx, event_file in enumerate(event_files, start=1):
            match_id = int(event_file.stem)
            with event_file.open("r", encoding="utf-8-sig") as f_in:
                records = json.load(f_in)

            if not isinstance(records, list):
                raise ValueError(f"Expected list of events in {event_file}")

            for record in records:
                row = process_record(record, match_id)
                out_row = {col: row.get(col, "") for col in ordered_columns}
                writer.writerow(out_row)
                total_rows += 1

                if report_sample_rows is None or total_rows <= report_sample_rows:
                    for col in REPORT_NULL_COLUMNS:
                        if is_null(out_row.get(col, "")):
                            null_counts[col] += 1

            logger.maybe_log(file_count=idx, row_count=total_rows)

    logger.maybe_log(file_count=len(event_files), row_count=total_rows, force=True)
    return total_rows, null_counts


def write_data_quality_report(
    report_path: Path,
    files_processed: int,
    total_rows: int,
    unique_match_count: int,
    missing_required_cols: list[str],
    null_counts: dict[str, int],
    available_columns: set[str],
    sampled_rows: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def pct(col: str) -> str:
        if col not in available_columns:
            return "column missing"
        if sampled_rows <= 0:
            return "not computed"
        return f"{(null_counts[col] / sampled_rows) * 100:.2f}%"

    lines = [
        "# Data Quality Report",
        "",
        "## Pipeline Summary",
        f"- Event files processed: {files_processed}",
        f"- Event rows written: {total_rows}",
        f"- Unique matches: {unique_match_count}",
        "",
        "## Required Columns Check",
    ]

    if missing_required_cols:
        lines.append("- Status: FAIL")
        lines.append(f"- Missing columns: {', '.join(missing_required_cols)}")
    else:
        lines.append("- Status: PASS")
        lines.append("- Missing columns: none")

    if sampled_rows != total_rows:
        lines.append(f"- Null-rate stats based on first {sampled_rows:,} rows")
    else:
        lines.append(f"- Null-rate stats based on all {sampled_rows:,} rows")

    lines.extend(
        [
            "",
            "## Stable ID Coverage",
            f"- `match_id` null rate: {pct('match_id')}",
            f"- `team_id` null rate: {pct('team_id')}",
            f"- `player_id` null rate: {pct('player_id')}",
            "",
            "## Coordinate Coverage",
            f"- `location_x` null rate: {pct('location_x')}",
            f"- `location_y` null rate: {pct('location_y')}",
            f"- `pass_end_location_x` null rate: {pct('pass_end_location_x')}",
            f"- `pass_end_location_y` null rate: {pct('pass_end_location_y')}",
            "",
            "## Notes",
            "- One row is produced per event record from source JSON arrays.",
            "- `match_id` is derived from source filenames.",
            "- Nested structures are flattened with underscore-separated columns.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)

    event_files = resolve_event_files(input_dir, args.max_files)
    if not event_files:
        raise ValueError("No events loaded from provided files.")

    all_columns, teams, players, pass1_rows, unique_matches = scan_schema_and_dimensions(
        event_files=event_files,
        log_every_files=args.log_every_files,
        log_every_rows=args.log_every_rows,
    )

    missing_required_cols = [c for c in REQUIRED_COLUMNS if c not in all_columns]
    if missing_required_cols:
        raise ValueError(f"Missing required columns after flattening: {missing_required_cols}")

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

    ordered_columns = [c for c in preferred if c in all_columns]
    ordered_columns += sorted(c for c in all_columns if c not in ordered_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    events_out = output_dir / "events.csv"
    teams_out = output_dir / "teams.csv"
    players_out = output_dir / "players.csv"

    rows_written, null_counts = stream_write_events_csv(
        event_files=event_files,
        output_path=events_out,
        ordered_columns=ordered_columns,
        log_every_files=args.log_every_files,
        log_every_rows=args.log_every_rows,
        report_sample_rows=args.report_sample_rows,
    )

    teams_rows = [
        {"team_id": team_id, "team_name": team_name}
        for team_id, team_name in sorted(teams.items(), key=lambda x: x[0])
    ]

    players_rows = [
        {
            "player_id": player_id,
            "player_name": player_name or "",
            "team_id": team_id if team_id is not None else "",
        }
        for player_id, (player_name, team_id) in sorted(players.items(), key=lambda x: x[0])
    ]

    write_csv(teams_out, teams_rows, ["team_id", "team_name"])
    write_csv(players_out, players_rows, ["player_id", "player_name", "team_id"])

    sampled_rows = rows_written
    if args.report_sample_rows is not None:
        sampled_rows = min(rows_written, max(0, args.report_sample_rows))

    write_data_quality_report(
        report_path=report_path,
        files_processed=len(event_files),
        total_rows=rows_written,
        unique_match_count=len(unique_matches),
        missing_required_cols=missing_required_cols,
        null_counts=null_counts,
        available_columns=all_columns,
        sampled_rows=sampled_rows,
    )

    print(f"Pass 1 rows scanned: {pass1_rows}")
    print(f"Processed {len(event_files)} files")
    print(f"Event rows written: {rows_written}")
    print(f"Wrote: {events_out}")
    print(f"Wrote: {teams_out}")
    print(f"Wrote: {players_out}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
