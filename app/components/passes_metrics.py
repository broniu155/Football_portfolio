from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PROGRESSIVE_THRESHOLD_DEFAULT = 10.0
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
LEFT_CHANNEL_MAX = PITCH_WIDTH / 3.0
RIGHT_CHANNEL_MIN = 2.0 * PITCH_WIDTH / 3.0


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def pass_completed_mask(pass_df: pd.DataFrame) -> tuple[pd.Series, bool]:
    if pass_df.empty:
        return pd.Series(dtype="bool"), False

    if "pass_outcome_name" in pass_df.columns:
        norm = _as_text(pass_df["pass_outcome_name"])
        complete = norm.isna() | norm.eq("") | norm.eq("complete") | norm.eq("none")
        return complete.fillna(False), True

    if "pass_outcome_id" in pass_df.columns:
        outcome_id = pd.to_numeric(pass_df["pass_outcome_id"], errors="coerce")
        complete = outcome_id.isna()
        return complete.fillna(False), True

    return pd.Series(True, index=pass_df.index), False


def progressive_pass_mask(pass_df: pd.DataFrame, threshold: float = PROGRESSIVE_THRESHOLD_DEFAULT) -> pd.Series:
    if not {"location_x", "pass_end_location_x"}.issubset(pass_df.columns):
        return pd.Series(False, index=pass_df.index)
    start_x = pd.to_numeric(pass_df["location_x"], errors="coerce")
    end_x = pd.to_numeric(pass_df["pass_end_location_x"], errors="coerce")
    start_dist = PITCH_LENGTH - start_x
    end_dist = PITCH_LENGTH - end_x
    gained = start_dist - end_dist
    return gained.ge(float(threshold)).fillna(False)


def attack_channel(end_y: pd.Series) -> pd.Series:
    y = pd.to_numeric(end_y, errors="coerce")
    out = pd.Series("Unknown", index=end_y.index, dtype="string")
    out = out.mask(y < LEFT_CHANNEL_MAX, "Left")
    out = out.mask((y >= LEFT_CHANNEL_MAX) & (y <= RIGHT_CHANNEL_MIN), "Centre")
    out = out.mask(y > RIGHT_CHANNEL_MIN, "Right")
    return out


def with_pass_features(pass_df: pd.DataFrame, threshold: float = PROGRESSIVE_THRESHOLD_DEFAULT) -> tuple[pd.DataFrame, bool]:
    out = pass_df.copy()
    completed, completion_available = pass_completed_mask(out)
    out["is_completed"] = completed.fillna(False).astype(bool)
    out["is_progressive"] = progressive_pass_mask(out, threshold=threshold).fillna(False).astype(bool)
    out["is_successful_progressive"] = (out["is_completed"] & out["is_progressive"]).astype(bool)
    if "pass_end_location_y" in out.columns:
        out["attack_channel"] = attack_channel(out["pass_end_location_y"])
    else:
        out["attack_channel"] = "Unknown"
    if {"location_x", "pass_end_location_x"}.issubset(out.columns):
        start_x = pd.to_numeric(out["location_x"], errors="coerce")
        end_x = pd.to_numeric(out["pass_end_location_x"], errors="coerce")
        out["progressive_distance_gained"] = (end_x - start_x).where(out["is_progressive"], 0.0).fillna(0.0)
    else:
        out["progressive_distance_gained"] = 0.0
    return out, completion_available


@dataclass(frozen=True)
class ChannelSummary:
    left: int
    centre: int
    right: int

    @property
    def total(self) -> int:
        return int(self.left + self.centre + self.right)


def summarize_channels(pass_df: pd.DataFrame) -> ChannelSummary:
    if pass_df.empty or "attack_channel" not in pass_df.columns:
        return ChannelSummary(left=0, centre=0, right=0)
    channel = pass_df["attack_channel"].astype("string")
    return ChannelSummary(
        left=int(channel.eq("Left").sum()),
        centre=int(channel.eq("Centre").sum()),
        right=int(channel.eq("Right").sum()),
    )


def top_progressive_passers(pass_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if pass_df.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "successful_progressive", "progressive_attempts", "progressive_completion_pct", "avg_progressive_gain"])
    work = pass_df.copy()
    if "player_name" not in work.columns:
        work["player_name"] = work.get("player_id", pd.Series(index=work.index)).astype("string")
    group_cols = [c for c in ("player_id", "player_name") if c in work.columns]
    if not group_cols:
        return pd.DataFrame(columns=["player_name", "successful_progressive", "progressive_attempts", "progressive_completion_pct", "avg_progressive_gain"])

    progressive_attempt = work["is_progressive"].fillna(False)
    successful = work["is_successful_progressive"].fillna(False)
    agg = (
        work.assign(
            progressive_attempt=progressive_attempt.astype(int),
            successful_progressive=successful.astype(int),
        )
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(
            successful_progressive=("successful_progressive", "sum"),
            progressive_attempts=("progressive_attempt", "sum"),
            avg_progressive_gain=("progressive_distance_gained", "mean"),
        )
    )
    agg["progressive_completion_pct"] = (
        (agg["successful_progressive"] / agg["progressive_attempts"].where(agg["progressive_attempts"] > 0, 1)) * 100.0
    ).round(1)
    agg.loc[agg["progressive_attempts"] == 0, "progressive_completion_pct"] = 0.0
    agg["avg_progressive_gain"] = pd.to_numeric(agg["avg_progressive_gain"], errors="coerce").fillna(0.0).round(1)
    agg = agg.sort_values(["successful_progressive", "progressive_completion_pct", "avg_progressive_gain"], ascending=[False, False, False])
    return agg.head(int(top_n)).reset_index(drop=True)

