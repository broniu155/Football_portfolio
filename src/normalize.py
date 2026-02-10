from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_X_COLUMNS = [
    "location_x",
    "pass_end_location_x",
    "carry_end_location_x",
    "shot_end_location_x",
]

DEFAULT_Y_COLUMNS = [
    "location_y",
    "pass_end_location_y",
    "carry_end_location_y",
    "shot_end_location_y",
]


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_coordinate(value: Any, pitch_dimension: float) -> float | None:
    """Normalize a single coordinate value from StatsBomb scale to 0-100."""
    numeric = _to_float(value)
    if numeric is None:
        return None
    return clamp((numeric / pitch_dimension) * 100.0)


def normalize_event_coordinates(
    row: dict[str, Any],
    home_team_id: int,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
    team_id_col: str = "team_id",
    x_columns: list[str] | None = None,
    y_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Return a normalized copy of one event row.

    X and Y are scaled to 0-100. X is flipped for non-home teams so both
    home and away attacks are oriented in the same direction.
    """
    x_columns = x_columns or DEFAULT_X_COLUMNS
    y_columns = y_columns or DEFAULT_Y_COLUMNS

    out = dict(row)
    team_id = _to_int(row.get(team_id_col))
    flip_x = team_id is not None and team_id != home_team_id

    for col in x_columns:
        if col not in row:
            continue
        norm = normalize_coordinate(row.get(col), pitch_length)
        if norm is None:
            out[col] = ""
            continue
        if flip_x:
            norm = 100.0 - norm
        out[col] = round(clamp(norm), 6)

    for col in y_columns:
        if col not in row:
            continue
        norm = normalize_coordinate(row.get(col), pitch_width)
        if norm is None:
            out[col] = ""
            continue
        out[col] = round(clamp(norm), 6)

    return out


def normalize_events(
    rows: list[dict[str, Any]],
    home_team_id: int,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
    team_id_col: str = "team_id",
    x_columns: list[str] | None = None,
    y_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize coordinates for a list of event rows."""
    return [
        normalize_event_coordinates(
            row=row,
            home_team_id=home_team_id,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            team_id_col=team_id_col,
            x_columns=x_columns,
            y_columns=y_columns,
        )
        for row in rows
    ]


def validate_coordinate_bounds(
    rows: list[dict[str, Any]],
    x_columns: list[str] | None = None,
    y_columns: list[str] | None = None,
) -> list[str]:
    """Return validation issues for coordinates outside 0-100."""
    x_columns = x_columns or DEFAULT_X_COLUMNS
    y_columns = y_columns or DEFAULT_Y_COLUMNS
    issues: list[str] = []

    for idx, row in enumerate(rows):
        for col in x_columns + y_columns:
            if col not in row:
                continue
            value = _to_float(row.get(col))
            if value is None:
                continue
            if value < 0.0 or value > 100.0:
                issues.append(f"row {idx}: {col} out of bounds ({value})")

    return issues


def normalize_csv(
    input_csv: Path,
    output_csv: Path,
    home_team_id: int,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> tuple[int, list[str]]:
    """Normalize event coordinates from CSV and write normalized CSV output."""
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    normalized_rows = normalize_events(
        rows=rows,
        home_team_id=home_team_id,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    issues = validate_coordinate_bounds(normalized_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return len(normalized_rows), issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize StatsBomb pitch coordinates to 0-100 and align attack direction."
    )
    parser.add_argument("--input-csv", required=True, help="Path to events CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to normalized output CSV.")
    parser.add_argument("--home-team-id", required=True, type=int, help="Home team ID for direction alignment.")
    parser.add_argument("--pitch-length", default=120.0, type=float, help="Source pitch length.")
    parser.add_argument("--pitch-width", default=80.0, type=float, help="Source pitch width.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_written, issues = normalize_csv(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        home_team_id=args.home_team_id,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
    )

    print(f"Rows written: {rows_written}")
    if issues:
        print(f"Validation issues: {len(issues)}")
        for issue in issues[:10]:
            print(f"- {issue}")
        raise ValueError("Found normalized coordinates outside 0-100")

    print("Validation passed: all normalized coordinates are within 0-100")


if __name__ == "__main__":
    main()
