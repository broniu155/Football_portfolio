from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUCKET_CONFIG = REPO_ROOT / "app" / "assets" / "event_buckets.yml"


class EventBucket(str, Enum):
    OFFENSIVE = "OFFENSIVE"
    DEFENSIVE = "DEFENSIVE"
    TRANSITION = "TRANSITION"
    SET_PIECE = "SET_PIECE"
    OTHER = "OTHER"


SET_PIECE_PATTERNS = {
    "from corner",
    "from free kick",
    "from throw in",
    "from goal kick",
    "from kick off",
    "from keeper",
    "from kick-off",
    "from throw-in",
    "from set piece",
}
COUNTER_PATTERN = "from counter"
TURNOVER_TYPES = {"miscontrol", "dispossessed", "error"}
REGAIN_TYPES = {"interception", "ball recovery", "block", "clearance", "duel", "50/50"}
DUEL_WIN_TOKENS = {"won", "win", "success", "success in play", "success out"}
BALL_RECEIPT_TYPES = {"ball receipt", "ball receipt*"}


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _as_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype).lower() in {"bool", "boolean"}:
        return series.fillna(False).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    return text.isin({"true", "t", "1", "yes", "y"}) | numeric.eq(1)


def _coalesce(df: pd.DataFrame, candidates: list[str], default: Any = pd.NA) -> pd.Series:
    existing = [column for column in candidates if column in df.columns]
    if not existing:
        return pd.Series(default, index=df.index, dtype="object")
    out = df[existing[0]]
    for column in existing[1:]:
        out = out.combine_first(df[column])
    return out


def _parse_simple_yaml_mapping(config_path: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {"offensive": [], "defensive": [], "transition": []}
    if not config_path.exists():
        return mapping
    current_key: str | None = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1].strip().lower()
            if key in mapping:
                current_key = key
            else:
                current_key = None
            continue
        stripped = line.strip()
        if current_key and stripped.startswith("- "):
            mapping[current_key].append(stripped[2:].strip().strip('"').strip("'"))
    return mapping


def load_bucket_mapping(config_path: Path | None = None) -> dict[str, str]:
    path = config_path or DEFAULT_BUCKET_CONFIG
    loaded: dict[str, list[str]]
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        loaded = {
            "offensive": list((payload or {}).get("offensive", []) or []),
            "defensive": list((payload or {}).get("defensive", []) or []),
            "transition": list((payload or {}).get("transition", []) or []),
        }
    except Exception:
        loaded = _parse_simple_yaml_mapping(path)

    mapping: dict[str, str] = {}
    for name in loaded.get("offensive", []):
        mapping[str(name).strip().lower()] = EventBucket.OFFENSIVE.value
    for name in loaded.get("defensive", []):
        mapping[str(name).strip().lower()] = EventBucket.DEFENSIVE.value
    for name in loaded.get("transition", []):
        mapping[str(name).strip().lower()] = EventBucket.TRANSITION.value
    return mapping


def _turnover_mask(event_types: pd.Series, df: pd.DataFrame) -> pd.Series:
    mask = event_types.isin(TURNOVER_TYPES)
    receipt_outcome = _coalesce(
        df,
        candidates=[
            "ball_receipt_outcome_name",
            "ball_receipt_outcome",
            "ball_receipt.outcome.name",
            "outcome_name",
        ],
    )
    incomplete_receipt = event_types.isin(BALL_RECEIPT_TYPES) & _normalize_text(receipt_outcome).eq("incomplete")
    return mask | incomplete_receipt.fillna(False)


def _regain_mask(event_types: pd.Series, df: pd.DataFrame) -> pd.Series:
    base = event_types.isin(REGAIN_TYPES)
    duel_outcome = _coalesce(df, ["duel_outcome_name", "duel_outcome", "fifty_fifty_outcome_name", "50_50_outcome_name"])
    duel_won = event_types.isin({"duel", "50/50"}) & _normalize_text(duel_outcome).isin(DUEL_WIN_TOKENS)
    return base | duel_won.fillna(False)


def _canonical_pass_height(df: pd.DataFrame) -> pd.Series:
    value = _coalesce(df, ["pass_height_name", "pass_height"])
    return value.astype("string").str.strip()


def _canonical_pass_outcome(df: pd.DataFrame) -> pd.Series:
    value = _coalesce(df, ["pass_outcome_name", "pass_outcome"])
    return value.astype("string").str.strip()


def derive_event_labels(events_df: pd.DataFrame, config_path: Path | None = None) -> pd.DataFrame:
    if events_df.empty:
        out = events_df.copy()
        for col in (
            "bucket",
            "subtype",
            "is_under_pressure",
            "is_counterpress",
            "is_set_piece",
            "is_counter",
            "is_turnover",
            "is_regain",
            "pass_height",
            "pass_outcome",
        ):
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out

    out = events_df.copy()
    type_col = "type_name" if "type_name" in out.columns else ("type" if "type" in out.columns else None)
    if type_col is None:
        event_types = pd.Series("", index=out.index, dtype="string")
    else:
        event_types = _normalize_text(out[type_col])

    play_pattern = _normalize_text(_coalesce(out, ["play_pattern_name", "play_pattern"]))

    is_under_pressure = _as_bool(out["under_pressure"]) if "under_pressure" in out.columns else pd.Series(False, index=out.index)
    is_counterpress = _as_bool(out["counterpress"]) if "counterpress" in out.columns else pd.Series(False, index=out.index)
    is_set_piece = play_pattern.isin(SET_PIECE_PATTERNS)
    is_counter = play_pattern.eq(COUNTER_PATTERN)
    is_turnover = _turnover_mask(event_types, out)
    is_regain = _regain_mask(event_types, out)

    bucket_map = load_bucket_mapping(config_path=config_path)
    base_bucket = event_types.map(bucket_map).fillna(EventBucket.OTHER.value)
    is_transition = is_turnover | is_counterpress | is_counter
    bucket = pd.Series(base_bucket, index=out.index, dtype="object")
    bucket = bucket.mask(is_transition, EventBucket.TRANSITION.value)
    bucket = bucket.mask(is_set_piece, EventBucket.SET_PIECE.value)

    subtype = _coalesce(out, ["type_name", "type"]).astype("string").str.strip()
    subtype = subtype.mask(is_turnover, "Turnover")

    out["bucket"] = bucket.astype("string")
    out["subtype"] = subtype
    out["is_under_pressure"] = is_under_pressure.fillna(False).astype(bool)
    out["is_counterpress"] = is_counterpress.fillna(False).astype(bool)
    out["is_set_piece"] = is_set_piece.fillna(False).astype(bool)
    out["is_counter"] = is_counter.fillna(False).astype(bool)
    out["is_turnover"] = is_turnover.fillna(False).astype(bool)
    out["is_regain"] = is_regain.fillna(False).astype(bool)
    out["pass_height"] = _canonical_pass_height(out)
    out["pass_outcome"] = _canonical_pass_outcome(out)
    return out


def _event_time_seconds(events_df: pd.DataFrame) -> pd.Series | None:
    if {"minute", "second"}.issubset(events_df.columns):
        minute = pd.to_numeric(events_df["minute"], errors="coerce").fillna(0)
        second = pd.to_numeric(events_df["second"], errors="coerce").fillna(0)
        return minute * 60 + second
    if "timestamp" in events_df.columns:
        ts = pd.to_timedelta(events_df["timestamp"], errors="coerce")
        if ts.notna().any():
            return ts.dt.total_seconds()
    return None


def derive_counterpress_regains(events_df: pd.DataFrame, window_seconds: float = 6.0) -> pd.DataFrame:
    if events_df.empty:
        out = events_df.copy()
        if "is_counterpress_regain" not in out.columns:
            out["is_counterpress_regain"] = pd.Series(dtype="bool")
        return out
    if "team_id" not in events_df.columns:
        out = events_df.copy()
        out["is_counterpress_regain"] = False
        return out

    out = events_df.copy()
    if "is_turnover" not in out.columns or "is_regain" not in out.columns:
        out = derive_event_labels(out)

    order_col = "event_index" if "event_index" in out.columns else ("index" if "index" in out.columns else None)
    if order_col:
        out = out.sort_values(by=order_col).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    time_seconds = _event_time_seconds(out)
    if time_seconds is None:
        out["is_counterpress_regain"] = False
        return out

    out["is_counterpress_regain"] = False
    turnovers = out.index[out["is_turnover"].fillna(False)].tolist()
    if not turnovers:
        return out

    team_ids = pd.to_numeric(out["team_id"], errors="coerce")
    for turnover_idx in turnovers:
        losing_team = team_ids.iloc[turnover_idx]
        if pd.isna(losing_team):
            continue
        t0 = float(time_seconds.iloc[turnover_idx])
        if pd.isna(t0):
            continue
        future = out.index[out.index > turnover_idx]
        if len(future) == 0:
            continue
        dt = time_seconds.loc[future] - t0
        within_window = dt[(dt >= 0) & (dt <= float(window_seconds))].index
        if len(within_window) == 0:
            continue
        regains = within_window[
            (out.loc[within_window, "is_regain"].fillna(False))
            & (pd.to_numeric(out.loc[within_window, "team_id"], errors="coerce") == losing_team)
        ]
        if len(regains) > 0:
            out.loc[regains[0], "is_counterpress_regain"] = True
    return out
