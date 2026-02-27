from __future__ import annotations

import io

import pandas as pd

from app.components.analysis_registry import classify_analysis_groups
from app.components.attack_channels import derive_attack_channel_columns
from app.components.dribbles import prepare_dribble_events
from app.components.passes_metrics import PROGRESSIVE_THRESHOLD_DEFAULT, pass_completed_mask, progressive_pass_mask

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
    "analysis_group",
    "analysis_subgroup",
    "pass_end_location_x",
    "pass_end_location_y",
    "pass_outcome_name",
    "pass_outcome_id",
    "shot_outcome_name",
    "shot_outcome_id",
    "shot_statsbomb_xg",
    "dribble_outcome_name",
    "dribble_is_attempt",
    "dribble_is_complete",
    "dribble_is_incomplete",
    "dribble_outcome_raw",
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
    out = derive_attack_channel_columns(df.copy())
    event_type = _coalesce(out, ["type_name"]).astype("string").str.strip().str.lower()
    out["is_shot"] = event_type.eq("shot")
    out["is_pass"] = event_type.eq("pass")
    out["is_duel"] = event_type.eq("duel")
    out["is_foul"] = event_type.isin({"foul committed", "foul won"})

    completed, _ = pass_completed_mask(out)
    out["pass_completed"] = completed.fillna(False).astype(bool)

    out["progressive_pass"] = progressive_pass_mask(out, threshold=PROGRESSIVE_THRESHOLD_DEFAULT).fillna(False).astype(bool)

    shot_outcome = _coalesce(out, ["shot_outcome_name", "shot_outcome"]).astype("string").str.strip().str.lower()
    out["shot_is_goal"] = shot_outcome.eq("goal")
    return out


def _with_dribble_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    event_type = _coalesce(out, ["type_name"]).astype("string").str.strip().str.lower()
    dribble_outcome_name = _coalesce(out, ["dribble_outcome_name", "dribble_outcome"], default=pd.NA).astype("string").str.strip()
    dribble_outcome_id = _coalesce(out, ["dribble_outcome_id"], default=pd.NA).astype("string").str.strip()
    out["dribble_outcome_name"] = dribble_outcome_name
    out["dribble_outcome_raw"] = pd.NA
    has_raw = dribble_outcome_name.notna() | dribble_outcome_id.notna()
    out.loc[has_raw, "dribble_outcome_raw"] = (
        "id=" + dribble_outcome_id.fillna("") + "|name=" + dribble_outcome_name.fillna("")
    ).loc[has_raw]
    out["dribble_is_attempt"] = event_type.eq("dribble")
    out["dribble_is_complete"] = False
    out["dribble_is_incomplete"] = False

    dribbles, _ = prepare_dribble_events(out)
    if dribbles.empty or "event_id" not in dribbles.columns or "event_id" not in out.columns:
        return out

    if "match_id" in dribbles.columns and "match_id" in out.columns:
        join_cols = ["match_id", "event_id"]
        dribble_flags = dribbles[join_cols + ["dribble_is_complete", "dribble_is_incomplete"]].copy()
    else:
        join_cols = ["event_id"]
        dribble_flags = dribbles[join_cols + ["dribble_is_complete", "dribble_is_incomplete"]].copy()
    out = out.merge(
        dribble_flags,
        on=join_cols,
        how="left",
        suffixes=("", "_resolved"),
    )
    out["dribble_is_complete"] = out["dribble_is_complete_resolved"].astype("boolean").fillna(False).astype(bool)
    out["dribble_is_incomplete"] = out["dribble_is_incomplete_resolved"].astype("boolean").fillna(False).astype(bool)
    return out.drop(columns=[c for c in ("dribble_is_complete_resolved", "dribble_is_incomplete_resolved") if c in out.columns])


def build_match_events_export_df(events_df: pd.DataFrame, include_derived: bool, essential_only: bool) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()

    out = events_df.copy()
    out = classify_analysis_groups(out)
    out = _with_dribble_columns(out)
    if include_derived:
        out = _derived_columns(out)

    if essential_only:
        cols = [c for c in ESSENTIAL_COLUMNS if c in out.columns]
        if include_derived:
            cols.extend(
                [
                    c
                    for c in [
                        "is_shot",
                        "is_pass",
                        "is_duel",
                        "is_foul",
                        "pass_completed",
                        "progressive_pass",
                        "attack_channel",
                        "channel_source",
                        "channel_reason",
                        "x_used",
                        "y_used",
                        "shot_is_goal",
                        "dribble_outcome_name",
                        "dribble_is_attempt",
                        "dribble_is_complete",
                        "dribble_is_incomplete",
                        "dribble_outcome_raw",
                        "analysis_group",
                        "analysis_subgroup",
                    ]
                    if c in out.columns
                ]
            )
        if cols:
            out = out[cols]

    return out


def events_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
