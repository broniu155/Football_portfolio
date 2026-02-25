from __future__ import annotations

import io

import pandas as pd

from app.components.passes_metrics import PROGRESSIVE_THRESHOLD_DEFAULT, attack_channel, pass_completed_mask, progressive_pass_mask

ESSENTIAL_COLUMNS = [
    "event_id",
    "match_id",
    "event_index",
    "index",
    "period",
    "timestamp",
    "minute",
    "second",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "position_name",
    "type_id",
    "type_name",
    "play_pattern_name",
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
    "pass_outcome_name",
    "pass_outcome_id",
    "shot_outcome_name",
    "shot_outcome_id",
    "shot_statsbomb_xg",
]


def _coalesce(df: pd.DataFrame, cols: list[str], default: object = pd.NA) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(default, index=df.index, dtype="object")
    out = df[existing[0]]
    for col in existing[1:]:
        out = out.combine_first(df[col])
    return out


def _derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    event_type = _coalesce(out, ["type_name"]).astype("string").str.strip().str.lower()
    out["is_shot"] = event_type.eq("shot")
    out["is_pass"] = event_type.eq("pass")
    out["is_duel"] = event_type.eq("duel")
    out["is_foul"] = event_type.isin({"foul committed", "foul won"})

    completed, _ = pass_completed_mask(out)
    out["pass_completed"] = completed.fillna(False).astype(bool)

    out["progressive_pass"] = progressive_pass_mask(out, threshold=PROGRESSIVE_THRESHOLD_DEFAULT).fillna(False).astype(bool)
    pass_end_y = _coalesce(out, ["pass_end_location_y"])
    out["attack_channel"] = attack_channel(pass_end_y)

    shot_outcome = _coalesce(out, ["shot_outcome_name", "shot_outcome"]).astype("string").str.strip().str.lower()
    out["shot_is_goal"] = shot_outcome.eq("goal")
    return out


def build_match_events_export_df(events_df: pd.DataFrame, include_derived: bool, essential_only: bool) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()

    out = events_df.copy()
    if include_derived:
        out = _derived_columns(out)

    if essential_only:
        cols = [c for c in ESSENTIAL_COLUMNS if c in out.columns]
        if include_derived:
            cols.extend([c for c in ["is_shot", "is_pass", "is_duel", "is_foul", "pass_completed", "progressive_pass", "attack_channel", "shot_is_goal"] if c in out.columns])
        if cols:
            out = out[cols]

    return out


def events_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
