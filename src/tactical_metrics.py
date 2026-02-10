from __future__ import annotations

from typing import Iterable

import pandas as pd


def _require_columns(df: pd.DataFrame, required: Iterable[str], fn_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{fn_name} requires columns: {missing}")


def _prepare_events_for_ordering(events: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["match_id", "period", "minute", "second", "event_index", "team_id"]
    _require_columns(events, base_cols, "_prepare_events_for_ordering")

    out = events.copy()
    for col in ["period", "minute", "second", "event_index", "team_id", "match_id"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values(
        ["match_id", "period", "minute", "second", "event_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def _base_event_view(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "match_id",
        "event_id",
        "team_id",
        "player_id",
        "type_name",
        "minute",
        "second",
        "location_x",
        "location_y",
        "pass_end_location_x",
        "pass_end_location_y",
    ]
    available = [c for c in cols if c in events.columns]
    out = events[available].copy()

    for col in [
        "match_id",
        "team_id",
        "player_id",
        "minute",
        "second",
        "location_x",
        "location_y",
        "pass_end_location_x",
        "pass_end_location_y",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def possessions(events: pd.DataFrame, turnover_gap_seconds: int = 10) -> pd.DataFrame:
    """Build one-row-per-possession segments using a simple sequence heuristic.

    Football logic:
    - A new possession begins when the team in control changes.
    - A new possession also begins after large dead-ball/time gaps (default: >10 sec).
    - A new possession starts at period boundaries.
    - Events are ordered by match, period, minute, second, event index.

    This does not require StatsBomb's native `possession` field and is designed
    to be transparent and reproducible for dashboard pipelines.
    """

    ordered = _prepare_events_for_ordering(events)
    ordered["abs_second"] = ordered["period"].fillna(1).sub(1).mul(45 * 60) + ordered[
        "minute"
    ].fillna(0).mul(60) + ordered["second"].fillna(0)

    ordered["prev_match"] = ordered["match_id"].shift(1)
    ordered["prev_team"] = ordered["team_id"].shift(1)
    ordered["prev_period"] = ordered["period"].shift(1)
    ordered["prev_abs_second"] = ordered["abs_second"].shift(1)

    new_possession = (
        (ordered["match_id"] != ordered["prev_match"])
        | (ordered["period"] != ordered["prev_period"])
        | (ordered["team_id"] != ordered["prev_team"])
        | ((ordered["abs_second"] - ordered["prev_abs_second"]) > turnover_gap_seconds)
    )

    ordered["possession_id"] = new_possession.groupby(ordered["match_id"]).cumsum()

    grouped = ordered.groupby(["match_id", "possession_id"], dropna=False, as_index=False)
    out = grouped.agg(
        team_id=("team_id", "first"),
        start_period=("period", "first"),
        start_minute=("minute", "first"),
        start_second=("second", "first"),
        end_period=("period", "last"),
        end_minute=("minute", "last"),
        end_second=("second", "last"),
        event_count=("event_id", "count"),
        start_x=("location_x", "first"),
        start_y=("location_y", "first"),
        end_x=("location_x", "last"),
        end_y=("location_y", "last"),
    )

    out["duration_seconds"] = (
        (out["end_minute"] * 60 + out["end_second"])
        - (out["start_minute"] * 60 + out["start_second"])
    ).clip(lower=0)
    return out


def passes_into_final_third(events: pd.DataFrame, final_third_x: float = 80.0) -> pd.DataFrame:
    """Return one row per completed pass ending in the attacking final third.

    Football logic:
    - Keeps only pass events.
    - Treats passes with missing `pass_outcome_name` as completed.
    - Counts entries where the pass ends in x >= `final_third_x` on a 120x80 pitch.
    - Requires pass to start outside final third (x < `final_third_x`).
    """

    _require_columns(
        events,
        ["type_name", "location_x", "pass_end_location_x", "event_id", "match_id", "team_id"],
        "passes_into_final_third",
    )
    out = _base_event_view(events)
    if "pass_outcome_name" in events.columns:
        complete_mask = events["pass_outcome_name"].isna() | (events["pass_outcome_name"] == "")
    else:
        complete_mask = pd.Series(True, index=events.index)

    mask = (
        (events["type_name"] == "Pass")
        & complete_mask
        & (pd.to_numeric(events["location_x"], errors="coerce") < final_third_x)
        & (pd.to_numeric(events["pass_end_location_x"], errors="coerce") >= final_third_x)
    )
    out = out.loc[mask].copy()
    out["metric"] = "pass_into_final_third"
    return out.reset_index(drop=True)


def progressive_passes(
    events: pd.DataFrame,
    own_half_x: float = 60.0,
    own_half_threshold: float = 15.0,
    cross_half_threshold: float = 10.0,
    opp_half_threshold: float = 10.0,
) -> pd.DataFrame:
    """Return one row per progressive completed pass.

    Football logic (left-to-right attacking orientation):
    - Progressive means a completed forward pass that moves the ball
      meaningfully toward goal.
    - Thresholds:
      - start/end in own half: >= 15 units forward
      - starts own half and ends opp half: >= 10 units forward
      - start/end in opp half: >= 10 units forward
    """

    _require_columns(
        events,
        ["type_name", "location_x", "pass_end_location_x", "event_id", "match_id", "team_id"],
        "progressive_passes",
    )
    out = _base_event_view(events)
    start_x = pd.to_numeric(events["location_x"], errors="coerce")
    end_x = pd.to_numeric(events["pass_end_location_x"], errors="coerce")
    delta_x = end_x - start_x

    if "pass_outcome_name" in events.columns:
        complete_mask = events["pass_outcome_name"].isna() | (events["pass_outcome_name"] == "")
    else:
        complete_mask = pd.Series(True, index=events.index)

    pass_mask = (events["type_name"] == "Pass") & complete_mask & (delta_x > 0)
    own_to_own = pass_mask & (start_x < own_half_x) & (end_x < own_half_x) & (delta_x >= own_half_threshold)
    own_to_opp = pass_mask & (start_x < own_half_x) & (end_x >= own_half_x) & (delta_x >= cross_half_threshold)
    opp_to_opp = pass_mask & (start_x >= own_half_x) & (end_x >= own_half_x) & (delta_x >= opp_half_threshold)

    mask = own_to_own | own_to_opp | opp_to_opp
    out = out.loc[mask].copy()
    out["progressive_distance"] = delta_x.loc[mask]
    out["metric"] = "progressive_pass"
    return out.reset_index(drop=True)


def box_entries(
    events: pd.DataFrame,
    box_start_x: float = 102.0,
    box_min_y: float = 18.0,
    box_max_y: float = 62.0,
) -> pd.DataFrame:
    """Return one row per completed pass or carry entering the penalty box.

    Football logic:
    - A box entry is a pass or carry that starts outside the box and ends inside:
      x >= box_start_x and box_min_y <= y <= box_max_y on a 120x80 pitch.
    - Passes require completion (missing `pass_outcome_name`).
    """

    _require_columns(
        events,
        ["event_id", "match_id", "team_id", "type_name", "location_x", "location_y"],
        "box_entries",
    )
    out = _base_event_view(events)

    start_x = pd.to_numeric(events["location_x"], errors="coerce")
    start_y = pd.to_numeric(events["location_y"], errors="coerce")

    pass_end_x = pd.to_numeric(events.get("pass_end_location_x"), errors="coerce")
    pass_end_y = pd.to_numeric(events.get("pass_end_location_y"), errors="coerce")
    carry_end_x = pd.to_numeric(events.get("carry_end_location_x"), errors="coerce")
    carry_end_y = pd.to_numeric(events.get("carry_end_location_y"), errors="coerce")

    if "pass_outcome_name" in events.columns:
        complete_pass = events["pass_outcome_name"].isna() | (events["pass_outcome_name"] == "")
    else:
        complete_pass = pd.Series(True, index=events.index)

    pass_entry = (
        (events["type_name"] == "Pass")
        & complete_pass
        & (start_x < box_start_x)
        & ~((start_y >= box_min_y) & (start_y <= box_max_y) & (start_x >= box_start_x))
        & (pass_end_x >= box_start_x)
        & (pass_end_y >= box_min_y)
        & (pass_end_y <= box_max_y)
    )

    carry_entry = (
        (events["type_name"] == "Carry")
        & (start_x < box_start_x)
        & ~((start_y >= box_min_y) & (start_y <= box_max_y) & (start_x >= box_start_x))
        & (carry_end_x >= box_start_x)
        & (carry_end_y >= box_min_y)
        & (carry_end_y <= box_max_y)
    )

    mask = pass_entry | carry_entry
    out = out.loc[mask].copy()
    out["entry_type"] = events.loc[mask, "type_name"].values
    out["metric"] = "box_entry"
    return out.reset_index(drop=True)


def defensive_actions_in_opposition_half(
    events: pd.DataFrame, halfway_x: float = 60.0
) -> pd.DataFrame:
    """Return one row per defensive action made in the opposition half.

    Football logic:
    - Defensive actions considered:
      Pressure, Duel, Interception, Block, Ball Recovery, Clearance, Foul Committed.
    - Opposition half condition (left-to-right orientation): event start x >= halfway.
    """

    _require_columns(
        events,
        ["event_id", "match_id", "team_id", "type_name", "location_x"],
        "defensive_actions_in_opposition_half",
    )
    out = _base_event_view(events)

    defensive_types = {
        "Pressure",
        "Duel",
        "Interception",
        "Block",
        "Ball Recovery",
        "Clearance",
        "Foul Committed",
    }

    mask = events["type_name"].isin(defensive_types) & (
        pd.to_numeric(events["location_x"], errors="coerce") >= halfway_x
    )
    out = out.loc[mask].copy()
    out["metric"] = "defensive_action_opposition_half"
    return out.reset_index(drop=True)

