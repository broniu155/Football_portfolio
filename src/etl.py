from __future__ import annotations

import argparse
import csv
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load StatsBomb event JSON files, flatten nested fields, "
            "write clean CSV outputs, and produce a data quality report."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data_raw/statsbomb",
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


def null_rate(rows: list[dict[str, Any]], col: str) -> float | None:
    if not rows or col not in rows[0]:
        return None

    null_count = sum(1 for row in rows if row.get(col) in (None, ""))
    return (null_count / len(rows)) * 100


def write_data_quality_report(
    report_path: Path,
    rows: list[dict[str, Any]],
    files_processed: int,
    missing_required_cols: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    unique_matches = {
        row.get("match_id") for row in rows if row.get("match_id") not in (None, "")
    }

    def pct(col: str) -> str:
        rate = null_rate(rows, col)
        return "column missing" if rate is None else f"{rate:.2f}%"

    lines = [
        "# Data Quality Report",
        "",
        "## Pipeline Summary",
        f"- Event files processed: {files_processed}",
        f"- Event rows written: {len(rows)}",
        f"- Unique matches: {len(unique_matches)}",
        "",
        "## Required Columns Check",
    ]

    if missing_required_cols:
        lines.append("- Status: FAIL")
        lines.append(f"- Missing columns: {', '.join(missing_required_cols)}")
    else:
        lines.append("- Status: PASS")
        lines.append("- Missing columns: none")

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

    all_rows: list[dict[str, Any]] = []
    all_columns: set[str] = set()
    teams: dict[int, str] = {}
    players: dict[int, tuple[str | None, int | None]] = {}

    for event_file in event_files:
        try:
            match_id = int(event_file.stem)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse match_id from filename: {event_file.name}. "
                "Expected numeric filename like 12345.json"
            ) from exc

        with event_file.open("r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected list of events in {event_file}")

        for record in records:
            row = process_record(record, match_id)
            all_rows.append(row)
            all_columns.update(row.keys())

            team_id = row.get("team_id")
            if team_id is not None:
                teams[team_id] = row.get("team_name") or ""

            player_id = row.get("player_id")
            if player_id is not None:
                players[player_id] = (row.get("player_name"), team_id)

    if not all_rows:
        raise ValueError("No events loaded from provided files.")

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

    events_rows = [{col: row.get(col, "") for col in ordered_columns} for row in all_rows]

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

    output_dir.mkdir(parents=True, exist_ok=True)
    events_out = output_dir / "events.csv"
    teams_out = output_dir / "teams.csv"
    players_out = output_dir / "players.csv"

    write_csv(events_out, events_rows, ordered_columns)
    write_csv(teams_out, teams_rows, ["team_id", "team_name"])
    write_csv(players_out, players_rows, ["player_id", "player_name", "team_id"])

    write_data_quality_report(
        report_path=report_path,
        rows=events_rows,
        files_processed=len(event_files),
        missing_required_cols=missing_required_cols,
    )

    print(f"Processed {len(event_files)} files")
    print(f"Event rows: {len(events_rows)}")
    print(f"Wrote: {events_out}")
    print(f"Wrote: {teams_out}")
    print(f"Wrote: {players_out}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
