from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd


def _team_filter(
    fact_events: pd.DataFrame,
    match_id: int,
    team_id: int | None = None,
    team_name: str | None = None,
) -> pd.DataFrame:
    if fact_events.empty:
        return fact_events.iloc[0:0].copy()

    work = fact_events.copy()
    if "match_id" in work.columns:
        work = work[pd.to_numeric(work["match_id"], errors="coerce") == int(match_id)]
    if team_id is not None and "team_id" in work.columns:
        work = work[pd.to_numeric(work["team_id"], errors="coerce") == int(team_id)]
    elif team_name and "team_name" in work.columns:
        work = work[work["team_name"].astype("string").str.strip().str.lower() == team_name.strip().lower()]
    return work


def _first_non_empty(series: pd.Series) -> str | None:
    for value in series.astype("string").fillna("").tolist():
        text = str(value).strip()
        if text:
            return text
    return None


def _parse_json_like(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_tactics_lineup(starting_row: pd.Series) -> list[dict[str, Any]]:
    for col in ("tactics_lineup", "lineup"):
        if col not in starting_row.index:
            continue
        parsed = _parse_json_like(starting_row.get(col))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("lineup"), list):
            return [item for item in parsed["lineup"] if isinstance(item, dict)]
    return []


def _shape_xi_from_tactics(lineup_items: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(lineup_items):
        player_block = item.get("player") if isinstance(item.get("player"), dict) else {}
        position_block = item.get("position") if isinstance(item.get("position"), dict) else {}
        rows.append(
            {
                "player_id": player_block.get("id"),
                "player_name": player_block.get("name"),
                "jersey_number": item.get("jersey_number"),
                "position_id": position_block.get("id"),
                "position_name": position_block.get("name"),
                "sort_order": idx + 1,
                "source": "starting_xi_event",
            }
        )
    return pd.DataFrame(rows)


def _canonical_xi_from_events(team_events: pd.DataFrame) -> pd.DataFrame:
    if team_events.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "jersey_number", "position_id", "position_name", "sort_order", "source"])

    work = team_events.copy()
    if "type_name" in work.columns:
        work = work[work["type_name"].astype("string").str.strip().str.lower() != "starting xi"]

    if "player_name" in work.columns:
        work = work[work["player_name"].astype("string").str.strip() != ""]
    if "player_id" in work.columns:
        work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce")

    for col in ("minute", "second", "period", "event_index"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    sort_cols = [col for col in ("period", "minute", "second", "event_index") if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort")

    out_rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()

    for _, row in work.iterrows():
        pid = row.get("player_id")
        pname = str(row.get("player_name") or "").strip()
        if pd.notna(pid):
            pid_i = int(pid)
            if pid_i in seen_ids:
                continue
        elif not pname:
            continue
        name_key = pname.lower()
        if name_key and name_key in seen_names:
            continue
        out_rows.append(
            {
                "player_id": int(pid) if pd.notna(pid) else pd.NA,
                "player_name": pname if pname else f"Player {len(out_rows) + 1}",
                "jersey_number": pd.NA,
                "position_id": row.get("position_id"),
                "position_name": row.get("position_name"),
                "sort_order": len(out_rows) + 1,
                "source": "events_first_appearance",
            }
        )
        if pd.notna(pid):
            seen_ids.add(int(pid))
        if name_key:
            seen_names.add(name_key)
        if len(out_rows) >= 11:
            break

    return pd.DataFrame(out_rows)


def _formation_from_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d(?:-\d+)+", text):
        return text
    if re.fullmatch(r"\d{3,5}", text):
        return "-".join(list(text))
    return None


def _position_bucket(position_name: str) -> str:
    p = position_name.lower()
    if "goalkeeper" in p or p == "gk":
        return "gk"
    if any(token in p for token in ("back", "defender", "center back", "left back", "right back", "wing back", "cb", "lb", "rb")):
        return "def"
    if any(token in p for token in ("forward", "striker", "wing", "attacker", "center forward", "cf", "st")):
        return "fwd"
    if any(token in p for token in ("midfield", "midfielder", "dm", "cm", "am")):
        return "mid"
    return "mid"


def _infer_formation_from_xi(xi_df: pd.DataFrame) -> str | None:
    if xi_df.empty:
        return None
    labels = xi_df.get("position_name", pd.Series("", index=xi_df.index)).astype("string").fillna("")
    buckets = labels.apply(_position_bucket)
    gk = int((buckets == "gk").sum())
    defenders = int((buckets == "def").sum())
    mids = int((buckets == "mid").sum())
    fwds = int((buckets == "fwd").sum())
    outfield = defenders + mids + fwds
    if outfield == 0:
        return None
    if gk == 0 and outfield >= 11:
        outfield = 10
    if outfield != 10:
        return None
    return f"{defenders}-{mids}-{fwds}"


def get_starting_xi(
    fact_events: pd.DataFrame,
    match_id: int,
    team_id: int | None = None,
    team_name: str | None = None,
) -> pd.DataFrame:
    team_events = _team_filter(fact_events, match_id=match_id, team_id=team_id, team_name=team_name)
    if team_events.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "jersey_number", "position_id", "position_name", "sort_order", "source"])

    starts = team_events[team_events.get("type_name", pd.Series("", index=team_events.index)).astype("string").str.strip().str.lower() == "starting xi"]
    if not starts.empty:
        lineup_items = _extract_tactics_lineup(starts.iloc[0])
        if lineup_items:
            xi = _shape_xi_from_tactics(lineup_items)
            if not xi.empty:
                return xi.drop_duplicates(subset=["player_id", "player_name"]).head(11).reset_index(drop=True)

    xi = _canonical_xi_from_events(team_events)
    return xi.drop_duplicates(subset=["player_id", "player_name"]).head(11).reset_index(drop=True)


def get_formation(
    fact_events: pd.DataFrame,
    match_id: int,
    team_id: int | None = None,
    team_name: str | None = None,
) -> str | None:
    team_events = _team_filter(fact_events, match_id=match_id, team_id=team_id, team_name=team_name)
    if team_events.empty:
        return None

    starts = team_events[team_events.get("type_name", pd.Series("", index=team_events.index)).astype("string").str.strip().str.lower() == "starting xi"]
    if not starts.empty:
        row = starts.iloc[0]
        for col in ("tactics_formation", "formation", "team_formation", "starting_formation"):
            if col in starts.columns:
                parsed = _formation_from_text(row.get(col))
                if parsed:
                    return parsed
        for col in ("tactics", "lineup"):
            parsed = _parse_json_like(row.get(col))
            if isinstance(parsed, dict):
                parsed_form = _formation_from_text(parsed.get("formation"))
                if parsed_form:
                    return parsed_form

    xi = get_starting_xi(team_events, match_id=match_id, team_id=team_id, team_name=team_name)
    inferred = _infer_formation_from_xi(xi)
    if inferred:
        return f"{inferred} (approx.)"
    return None


def _parse_formation_lines(formation: str | None) -> list[int]:
    if not formation:
        return [4, 3, 3]
    digits = [int(d) for d in re.findall(r"\d+", formation)]
    if not digits:
        return [4, 3, 3]
    if sum(digits) == 10:
        return digits
    if len(digits) == 1 and digits[0] >= 3:
        raw = [int(ch) for ch in str(digits[0])]
        if sum(raw) == 10:
            return raw
    return [4, 3, 3]


def get_starting_positions(
    fact_events: pd.DataFrame,
    match_id: int,
    team_id: int | None = None,
    team_name: str | None = None,
    formation: str | None = None,
) -> list[dict[str, Any]]:
    xi = get_starting_xi(fact_events, match_id=match_id, team_id=team_id, team_name=team_name)
    if xi.empty:
        return []

    lines = _parse_formation_lines(formation)
    remaining = xi.copy()
    remaining["position_name"] = remaining.get("position_name", pd.Series("", index=remaining.index)).astype("string").fillna("")

    gk_mask = remaining["position_name"].str.lower().str.contains("goalkeeper|\\bgk\\b", regex=True, na=False)
    if gk_mask.any():
        gk_row = remaining[gk_mask].iloc[0]
        outfield = remaining.drop(index=gk_row.name).reset_index(drop=True)
    else:
        gk_row = remaining.iloc[0]
        outfield = remaining.iloc[1:].reset_index(drop=True)

    needed = sum(lines)
    if len(outfield) < needed:
        pad = needed - len(outfield)
        fillers = pd.DataFrame(
            [
                {
                    "player_id": pd.NA,
                    "player_name": f"Unknown {i + 1}",
                    "jersey_number": pd.NA,
                    "position_id": pd.NA,
                    "position_name": "",
                    "sort_order": 100 + i,
                    "source": "fallback_padding",
                }
                for i in range(pad)
            ]
        )
        outfield = pd.concat([outfield, fillers], ignore_index=True)
    outfield = outfield.head(needed)

    x_rows: list[dict[str, Any]] = [
        {
            "player_id": gk_row.get("player_id"),
            "player_name": str(gk_row.get("player_name") or "Goalkeeper"),
            "jersey_number": gk_row.get("jersey_number"),
            "position_name": gk_row.get("position_name"),
            "x": 8.0,
            "y": 40.0,
            "approximate": True,
        }
    ]

    x_slots = np.linspace(26, 102, num=len(lines)).tolist()
    idx = 0
    for line_idx, count in enumerate(lines):
        if count <= 0:
            continue
        y_slots = np.linspace(14, 66, num=count).tolist()
        for y in y_slots:
            if idx >= len(outfield):
                break
            row = outfield.iloc[idx]
            idx += 1
            x_rows.append(
                {
                    "player_id": row.get("player_id"),
                    "player_name": str(row.get("player_name") or f"Player {idx}"),
                    "jersey_number": row.get("jersey_number"),
                    "position_name": row.get("position_name"),
                    "x": float(x_slots[line_idx]),
                    "y": float(y),
                    "approximate": True,
                }
            )
    return x_rows[:11]
