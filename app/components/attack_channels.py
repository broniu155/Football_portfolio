from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
LEFT_CHANNEL_MAX = PITCH_WIDTH / 3.0
RIGHT_CHANNEL_MIN = 2.0 * PITCH_WIDTH / 3.0


@dataclass(frozen=True)
class AttackChannelResult:
    channel: str | None
    reason: str


def compute_attack_channel(x: float | int | None, y: float | int | None) -> AttackChannelResult:
    """Compute lane from StatsBomb coordinates.

    Boundaries use half-open intervals for deterministic binning:
    - Left:   0 <= y < 80/3
    - Centre: 80/3 <= y < 160/3
    - Right:  160/3 <= y <= 80
    """
    x_num = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    y_num = pd.to_numeric(pd.Series([y]), errors="coerce").iloc[0]
    if pd.isna(x_num) or pd.isna(y_num):
        return AttackChannelResult(channel=None, reason="missing")
    if x_num < 0.0 or x_num > PITCH_LENGTH or y_num < 0.0 or y_num > PITCH_WIDTH:
        return AttackChannelResult(channel=None, reason="out_of_range")
    if y_num < LEFT_CHANNEL_MAX:
        return AttackChannelResult(channel="Left", reason="ok")
    if y_num < RIGHT_CHANNEL_MIN:
        return AttackChannelResult(channel="Centre", reason="ok")
    return AttackChannelResult(channel="Right", reason="ok")


def attack_channel_from_y(y: pd.Series, x: pd.Series | None = None) -> pd.Series:
    """Backward-compatible channel labels for pass views.

    Returns Left/Centre/Right/Unknown. Unknown is only used for missing or out-of-range
    coordinates when x is provided; without x, only y range is validated.
    """
    y_num = pd.to_numeric(y, errors="coerce")
    if x is None:
        x_num = pd.Series(1.0, index=y.index)
    else:
        x_num = pd.to_numeric(x, errors="coerce")

    valid = x_num.notna() & y_num.notna() & x_num.between(0.0, PITCH_LENGTH, inclusive="both") & y_num.between(0.0, PITCH_WIDTH, inclusive="both")
    out = pd.Series("Unknown", index=y.index, dtype="string")
    out.loc[valid & (y_num < LEFT_CHANNEL_MAX)] = "Left"
    out.loc[valid & (y_num >= LEFT_CHANNEL_MAX) & (y_num < RIGHT_CHANNEL_MIN)] = "Centre"
    out.loc[valid & (y_num >= RIGHT_CHANNEL_MIN)] = "Right"
    return out


def _coalesce(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    out = df[existing[0]]
    for col in existing[1:]:
        out = out.combine_first(df[col])
    return out


def _normalize_event_type(df: pd.DataFrame) -> pd.Series:
    if "type_name" not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df["type_name"].astype("string").str.strip().str.lower()


def _extract_xy_from_nested(series: pd.Series, idx: int) -> pd.Series:
    def _pick(value: object) -> float | None:
        if isinstance(value, (list, tuple)):
            if len(value) <= idx:
                return None
            return pd.to_numeric(value[idx], errors="coerce")
        if isinstance(value, str):
            txt = value.strip()
            if txt.startswith("[") and txt.endswith("]"):
                parts = [p.strip() for p in txt[1:-1].split(",")]
                if len(parts) <= idx:
                    return None
                return pd.to_numeric(parts[idx], errors="coerce")
        return None

    return pd.to_numeric(series.map(_pick), errors="coerce")


def derive_attack_channel_columns(events_df: pd.DataFrame) -> pd.DataFrame:
    """Derive attack channel and debug metadata from event/pass/carry/shot coordinates.

    Event applicability:
    - Works for all events with event.location.
    - Explicit fallback coverage for pass, carry, and shot rows.
    - Dribbles are covered via event.location when present.

    Source precedence:
    - Primary for all rows: event.location -> location_x/location_y
    - Fallback for event sub-types when primary is missing:
      pass.end_location, carry.end_location, shot.end_location
    """
    out = events_df.copy()
    event_type = _normalize_event_type(out)

    event_x = pd.to_numeric(_coalesce(out, ["location_x"]), errors="coerce")
    event_y = pd.to_numeric(_coalesce(out, ["location_y"]), errors="coerce")
    pass_x = pd.to_numeric(_coalesce(out, ["pass_end_location_x"]), errors="coerce")
    pass_y = pd.to_numeric(_coalesce(out, ["pass_end_location_y"]), errors="coerce")
    carry_x = pd.to_numeric(_coalesce(out, ["carry_end_location_x"]), errors="coerce")
    carry_y = pd.to_numeric(_coalesce(out, ["carry_end_location_y"]), errors="coerce")
    shot_x = pd.to_numeric(_coalesce(out, ["shot_end_location_x"]), errors="coerce")
    shot_y = pd.to_numeric(_coalesce(out, ["shot_end_location_y"]), errors="coerce")
    if "shot_end_location" in out.columns:
        nested = out["shot_end_location"]
        shot_x = shot_x.combine_first(_extract_xy_from_nested(nested, idx=0))
        shot_y = shot_y.combine_first(_extract_xy_from_nested(nested, idx=1))

    x_used = event_x.copy()
    y_used = event_y.copy()
    source = pd.Series("event.location", index=out.index, dtype="string")
    event_missing = x_used.isna() | y_used.isna()

    pass_mask = event_type.eq("pass") & event_missing & pass_x.notna() & pass_y.notna()
    carry_mask = event_type.eq("carry") & event_missing & carry_x.notna() & carry_y.notna()
    shot_mask = event_type.eq("shot") & event_missing & shot_x.notna() & shot_y.notna()

    x_used.loc[pass_mask] = pass_x.loc[pass_mask]
    y_used.loc[pass_mask] = pass_y.loc[pass_mask]
    source.loc[pass_mask] = "pass.end_location"

    x_used.loc[carry_mask] = carry_x.loc[carry_mask]
    y_used.loc[carry_mask] = carry_y.loc[carry_mask]
    source.loc[carry_mask] = "carry.end_location"

    x_used.loc[shot_mask] = shot_x.loc[shot_mask]
    y_used.loc[shot_mask] = shot_y.loc[shot_mask]
    source.loc[shot_mask] = "shot.end_location"

    still_missing = x_used.isna() | y_used.isna()
    source.loc[still_missing] = "none"

    in_range = x_used.between(0.0, PITCH_LENGTH, inclusive="both") & y_used.between(0.0, PITCH_WIDTH, inclusive="both")
    reason = pd.Series("ok", index=out.index, dtype="string")
    reason.loc[still_missing] = "missing"
    reason.loc[(~still_missing) & (~in_range)] = "out_of_range"

    channel = pd.Series("Unknown", index=out.index, dtype="string")
    valid = reason.eq("ok")
    channel.loc[valid & (y_used < LEFT_CHANNEL_MAX)] = "Left"
    channel.loc[valid & (y_used >= LEFT_CHANNEL_MAX) & (y_used < RIGHT_CHANNEL_MIN)] = "Centre"
    channel.loc[valid & (y_used >= RIGHT_CHANNEL_MIN)] = "Right"

    out["attack_channel"] = channel
    out["channel_source"] = source
    out["channel_reason"] = reason
    out["x_used"] = x_used
    out["y_used"] = y_used
    return out
