from __future__ import annotations

from typing import Any

import pandas as pd

ANALYSIS_VIEWS = (
    "Stats",
    "Offensive",
    "Passes",
    "Transitions",
    "Defensive",
    "Set Pieces",
    "More",
)

OFFENSIVE_COMPARISON_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("total_shots", "Total Shots", False),
    ("shots_on_target", "Shots on Target", False),
    ("shots_off_target", "Shots off Target", False),
    ("blocked_shots", "Blocked Shots", False),
    ("goals", "Goals", False),
    ("total_xg", "Total xG", False),
    ("xg_per_shot", "xG per Shot", False),
    ("dribble_attempts", "Dribble Attempts", False),
    ("dribbles_completed", "Dribbles Completed", False),
    ("dribble_success_pct", "Dribble Success %", True),
    ("shots_in_box", "Shots in Box", False),
    ("crosses", "Crosses", False),
)


def _coalesce(df: pd.DataFrame, candidates: list[str], default: Any = pd.NA) -> pd.Series:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series(default, index=df.index, dtype="object")
    out = df[cols[0]]
    for col in cols[1:]:
        out = out.combine_first(df[col])
    return out


def classify_analysis_groups(events_df: pd.DataFrame) -> pd.DataFrame:
    out = events_df.copy()
    if out.empty:
        out["analysis_group"] = pd.Series(dtype="string")
        out["analysis_subgroup"] = pd.Series(dtype="string")
        return out

    event_type = _coalesce(out, ["type_name", "type"], default="").astype("string").str.strip().str.lower()
    bucket = _coalesce(out, ["bucket"], default="").astype("string").str.strip().str.lower()
    duel_type = _coalesce(out, ["duel_type_name"], default="").astype("string").str.strip().str.lower()
    is_counter = _coalesce(out, ["is_counter"], default=False).astype("boolean").fillna(False).astype(bool)
    is_turnover = _coalesce(out, ["is_turnover"], default=False).astype("boolean").fillna(False).astype(bool)
    is_counterpress = _coalesce(out, ["is_counterpress"], default=False).astype("boolean").fillna(False).astype(bool)
    is_counterpress_regain = _coalesce(out, ["is_counterpress_regain"], default=False).astype("boolean").fillna(False).astype(bool)

    group = pd.Series("other", index=out.index, dtype="string")
    subgroup = pd.Series("other", index=out.index, dtype="string")

    pass_mask = event_type.eq("pass")
    shot_mask = event_type.eq("shot")
    carry_mask = event_type.eq("carry")
    dribble_mask = event_type.eq("dribble")
    duel_mask = event_type.isin({"duel", "50/50"})
    recoveries_mask = event_type.isin({"ball recovery", "interception"})
    defensive_action_mask = event_type.isin({"tackle", "block", "clearance", "pressure"})
    transition_mask = carry_mask | is_counter | is_turnover | is_counterpress | is_counterpress_regain | bucket.eq("transition")
    offensive_duel_mask = duel_mask & (duel_type.str.contains("offensive", na=False) | bucket.eq("offensive"))
    defensive_duel_mask = duel_mask & ~offensive_duel_mask
    foul_won_attacking_mask = event_type.eq("foul won") & bucket.eq("offensive")

    group.loc[pass_mask] = "passes"
    subgroup.loc[pass_mask] = "passes"

    group.loc[shot_mask] = "offensive"
    subgroup.loc[shot_mask] = "shots"

    group.loc[dribble_mask | offensive_duel_mask | foul_won_attacking_mask] = "offensive"
    subgroup.loc[dribble_mask | offensive_duel_mask | foul_won_attacking_mask] = "duels_offensive"

    group.loc[transition_mask] = "transitions"
    subgroup.loc[transition_mask] = "carries"
    subgroup.loc[transition_mask & ~carry_mask] = "transition_events"

    group.loc[recoveries_mask] = "defensive"
    subgroup.loc[recoveries_mask] = "recoveries"

    group.loc[defensive_duel_mask | defensive_action_mask] = "defensive"
    subgroup.loc[defensive_duel_mask | defensive_action_mask] = "duels_defensive"

    out["analysis_group"] = group
    out["analysis_subgroup"] = subgroup
    return out
