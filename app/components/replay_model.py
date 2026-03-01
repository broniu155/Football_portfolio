from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import streamlit as st

try:
    from app.components.data import (
        _query_fact_rows,
        _resolve_data_dir,
        _resolve_table_file,
        get_active_data_mode,
        load_match_events,
    )
except (ModuleNotFoundError, KeyError):
    from components.data import (  # type: ignore
        _query_fact_rows,
        _resolve_data_dir,
        _resolve_table_file,
        get_active_data_mode,
        load_match_events,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]

REPLAY_EVENT_COLUMNS = [
    "event_id",
    "match_id",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "type_name",
    "period",
    "minute",
    "second",
    "timestamp",
    "event_index",
    "index",
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
    "carry_end_location_x",
    "carry_end_location_y",
    "shot_end_location_x",
    "shot_end_location_y",
]


class ReplaySegment(TypedDict):
    t0: float
    t1: float
    period: int | None
    team: str
    team_id: int | None
    player: str
    player_id: int | None
    event_id: str
    event_type: str
    ball0: tuple[float, float] | None
    ball1: tuple[float, float] | None
    actor_xy: tuple[float, float] | None
    freeze_frame: list[dict[str, Any]]
    visible_area: list[list[float]]


def _to_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _period_offset_seconds(period: int | None) -> int:
    if period == 1:
        return 0
    if period == 2:
        return 45 * 60
    if period == 3:
        return 90 * 60
    if period == 4:
        return 105 * 60
    return 0


def parse_timestamp_seconds(timestamp: Any) -> float | None:
    """Convert StatsBomb event timestamp text to elapsed seconds in period."""
    if timestamp is None:
        return None
    text = str(timestamp).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = float(parts[2])
        return float(hh * 3600 + mm * 60) + ss
    except (TypeError, ValueError):
        return None


def period_timestamp_to_match_seconds(
    period: int | None,
    timestamp: Any,
    minute: Any = None,
    second: Any = None,
) -> float | None:
    """Convert period + timestamp fields to absolute match seconds."""
    base = _period_offset_seconds(period)
    ts = parse_timestamp_seconds(timestamp)
    if ts is not None:
        return base + ts

    mm = _to_int(minute)
    ss = _to_float(second)
    if mm is None or ss is None:
        return None

    period_offset_min = {1: 0, 2: 45, 3: 90, 4: 105}.get(period or 1, 0)
    local_min = mm - period_offset_min if mm >= period_offset_min else mm
    if local_min < 0:
        local_min = mm
    return float(base + local_min * 60 + ss)


def get_event_xy(event: dict[str, Any]) -> tuple[float, float] | None:
    """Extract event start location [x, y] from flattened or nested StatsBomb fields."""
    x = _to_float(event.get("location_x"))
    y = _to_float(event.get("location_y"))
    if x is not None and y is not None:
        return x, y

    location = event.get("location")
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        x = _to_float(location[0])
        y = _to_float(location[1])
        if x is not None and y is not None:
            return x, y
    return None


def get_event_end_xy(event: dict[str, Any]) -> tuple[float, float] | None:
    """Extract event end location from carry/pass/shot end_location fields."""
    pairs = [
        ("carry_end_location_x", "carry_end_location_y"),
        ("pass_end_location_x", "pass_end_location_y"),
        ("shot_end_location_x", "shot_end_location_y"),
    ]
    for x_col, y_col in pairs:
        x = _to_float(event.get(x_col))
        y = _to_float(event.get(y_col))
        if x is not None and y is not None:
            return x, y

    nested = [event.get("carry"), event.get("pass"), event.get("shot")]
    for obj in nested:
        if isinstance(obj, dict):
            end_location = obj.get("end_location")
            if isinstance(end_location, (list, tuple)) and len(end_location) >= 2:
                x = _to_float(end_location[0])
                y = _to_float(end_location[1])
                if x is not None and y is not None:
                    return x, y
    return None


@st.cache_data(show_spinner=False, ttl=600, max_entries=64)
def _load_events_df(match_id: int, data_mode: str) -> pd.DataFrame:
    return load_match_events(
        match_id=int(match_id),
        data_mode=str(data_mode),
        columns=REPLAY_EVENT_COLUMNS,
    )


def load_events(match_id: int, data_mode: str) -> list[dict[str, Any]]:
    """Load match events as list-of-dicts for replay normalization."""
    events = _load_events_df(match_id=int(match_id), data_mode=str(data_mode))
    return events.to_dict(orient="records") if not events.empty else []


def _parse_visible_area(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return []

    if not isinstance(raw, list) or len(raw) < 6:
        return []
    points: list[list[float]] = []
    for idx in range(0, len(raw), 2):
        if idx + 1 >= len(raw):
            break
        x = _to_float(raw[idx])
        y = _to_float(raw[idx + 1])
        if x is None or y is None:
            continue
        points.append([x, y])
    return points


def _load_visible_area_from_processed(match_id: int) -> pd.DataFrame:
    processed_dir = REPO_ROOT / "data_processed"
    processed_path = _resolve_table_file(processed_dir, "three_sixty_visible_area")
    if processed_path is None:
        return pd.DataFrame(columns=["event_uuid", "visible_area"])
    return _query_fact_rows(
        processed_path,
        match_id=int(match_id),
        selected_columns=["event_uuid", "visible_area", "visible_area_point_count"],
    )


def _load_from_raw_three_sixty_json(match_id: int) -> list[dict[str, Any]]:
    raw_path = REPO_ROOT / "data_raw" / "three-sixty" / f"{int(match_id)}.json"
    if not raw_path.exists():
        return []
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        event_uuid = str(row.get("event_uuid") or row.get("event_id") or row.get("id") or "").strip()
        if not event_uuid:
            continue
        freeze_frame_raw = row.get("freeze_frame")
        freeze_frame: list[dict[str, Any]] = []
        if isinstance(freeze_frame_raw, list):
            for ff in freeze_frame_raw:
                if not isinstance(ff, dict):
                    continue
                loc = ff.get("location")
                x = _to_float(ff.get("location_x"))
                y = _to_float(ff.get("location_y"))
                if (x is None or y is None) and isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    x = _to_float(loc[0])
                    y = _to_float(loc[1])
                if x is None or y is None:
                    continue
                freeze_frame.append(
                    {
                        "player_id": _to_int(ff.get("player_id")),
                        "location_x": x,
                        "location_y": y,
                        "teammate": _to_bool(ff.get("teammate")),
                        "actor": _to_bool(ff.get("actor")),
                        "keeper": _to_bool(ff.get("keeper")),
                    }
                )
        out.append(
            {
                "event_uuid": event_uuid,
                "freeze_frame": freeze_frame,
                "visible_area": _parse_visible_area(row.get("visible_area")),
            }
        )
    return out


@st.cache_data(show_spinner=False, ttl=600, max_entries=64)
def load_frames360(match_id: int, data_mode: str) -> list[dict[str, Any]] | None:
    """Load 360 freeze-frame rows joined by event_uuid. Returns None when unavailable."""
    mode = str(data_mode or get_active_data_mode()).strip().lower()
    base_dir = _resolve_data_dir(mode)
    ff_path = _resolve_table_file(base_dir, "fact_three_sixty_freeze_frames")
    vis_path = _resolve_table_file(base_dir, "fact_three_sixty_visible_area")

    by_event: dict[str, dict[str, Any]] = {}
    has_any = False

    if ff_path is not None:
        freeze_df = _query_fact_rows(
            ff_path,
            match_id=int(match_id),
            selected_columns=[
                "event_uuid",
                "player_id",
                "location_x",
                "location_y",
                "teammate",
                "actor",
                "keeper",
            ],
        )
        if not freeze_df.empty:
            has_any = True
            for row in freeze_df.to_dict(orient="records"):
                event_uuid = str(row.get("event_uuid") or "").strip()
                x = _to_float(row.get("location_x"))
                y = _to_float(row.get("location_y"))
                if not event_uuid or x is None or y is None:
                    continue
                bucket = by_event.setdefault(event_uuid, {"event_uuid": event_uuid, "freeze_frame": [], "visible_area": []})
                bucket["freeze_frame"].append(
                    {
                        "player_id": _to_int(row.get("player_id")),
                        "location_x": x,
                        "location_y": y,
                        "teammate": _to_bool(row.get("teammate")),
                        "actor": _to_bool(row.get("actor")),
                        "keeper": _to_bool(row.get("keeper")),
                    }
                )

    visible_frames = pd.DataFrame()
    if vis_path is not None:
        visible_frames = _query_fact_rows(
            vis_path,
            match_id=int(match_id),
            selected_columns=["event_uuid", "visible_area", "visible_area_point_count"],
        )
    elif mode == "local_generated":
        visible_frames = _load_visible_area_from_processed(match_id=int(match_id))

    if not visible_frames.empty:
        has_any = True
        for row in visible_frames.to_dict(orient="records"):
            event_uuid = str(row.get("event_uuid") or "").strip()
            if not event_uuid:
                continue
            bucket = by_event.setdefault(event_uuid, {"event_uuid": event_uuid, "freeze_frame": [], "visible_area": []})
            bucket["visible_area"] = _parse_visible_area(row.get("visible_area"))

    if not has_any:
        raw_payload = _load_from_raw_three_sixty_json(match_id=int(match_id))
        if raw_payload:
            return raw_payload
        return None

    return list(by_event.values())


@st.cache_data(show_spinner=False, ttl=600, max_entries=64)
def build_replay_segments(match_id: int, data_mode: str) -> tuple[list[ReplaySegment], dict[str, Any]]:
    """Build normalized replay segments from events and optional 360 snapshots."""
    events = load_events(match_id=int(match_id), data_mode=str(data_mode))
    if not events:
        return [], {"coverage": "No events", "has_360": False, "has_visible_area": False}

    frames = load_frames360(match_id=int(match_id), data_mode=str(data_mode))
    frames_by_uuid = {
        str(row.get("event_uuid") or "").strip(): row
        for row in (frames or [])
        if str(row.get("event_uuid") or "").strip()
    }

    rows: list[dict[str, Any]] = []
    for event in events:
        period = _to_int(event.get("period"))
        t_abs = period_timestamp_to_match_seconds(
            period=period,
            timestamp=event.get("timestamp"),
            minute=event.get("minute"),
            second=event.get("second"),
        )
        if t_abs is None:
            continue
        rows.append(
            {
                "event": event,
                "t_abs": float(t_abs),
                "event_index": _to_int(event.get("event_index")) or _to_int(event.get("index")) or 0,
            }
        )
    if not rows:
        return [], {"coverage": "No timestamped events", "has_360": False, "has_visible_area": False}

    rows.sort(key=lambda item: (item["t_abs"], item["event_index"]))
    segments: list[ReplaySegment] = []
    has_visible_area = False
    for idx, row in enumerate(rows):
        event = row["event"]
        t0 = float(row["t_abs"])
        t1 = float(rows[idx + 1]["t_abs"]) if idx + 1 < len(rows) else t0 + 2.0
        if t1 <= t0:
            t1 = t0 + 0.3

        ball0 = get_event_xy(event)
        ball1 = get_event_end_xy(event) or ball0
        if ball0 is None and ball1 is None:
            continue

        event_id = str(event.get("event_id") or event.get("event_uuid") or event.get("id") or "").strip()
        frame = frames_by_uuid.get(event_id, {})
        freeze_frame = frame.get("freeze_frame", []) if isinstance(frame, dict) else []
        visible_area = frame.get("visible_area", []) if isinstance(frame, dict) else []
        if isinstance(visible_area, list) and len(visible_area) >= 3:
            has_visible_area = True

        segment: ReplaySegment = {
            "t0": t0,
            "t1": t1,
            "period": _to_int(event.get("period")),
            "team": str(event.get("team_name") or event.get("team_id") or "Unknown"),
            "team_id": _to_int(event.get("team_id")),
            "player": str(event.get("player_name") or event.get("player_id") or "Unknown"),
            "player_id": _to_int(event.get("player_id")),
            "event_id": event_id,
            "event_type": str(event.get("type_name") or "Unknown"),
            "ball0": ball0,
            "ball1": ball1,
            "actor_xy": ball0,
            "freeze_frame": freeze_frame if isinstance(freeze_frame, list) else [],
            "visible_area": visible_area if isinstance(visible_area, list) else [],
        }
        segments.append(segment)

    has_360 = any(len(seg["freeze_frame"]) > 0 for seg in segments)
    coverage = "Events+360" if has_360 else "Events-only"
    return segments, {"coverage": coverage, "has_360": has_360, "has_visible_area": has_visible_area}
