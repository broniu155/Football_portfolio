from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
LINEUPS_DIR = REPO_ROOT / "data_raw" / "lineups"

# Canonical "attack-up" coordinates for top-half team on a 120x100 pitch.
# Home/away final placement is handled by explicit transform using `is_home`.
POSITION_COORDS_ATTACK_UP: dict[str, tuple[float, float]] = {
    "goalkeeper": (60.0, 92.0),
    "left back": (24.0, 82.0),
    "left center back": (42.0, 84.0),
    "center back": (60.0, 84.0),
    "right center back": (78.0, 84.0),
    "right back": (96.0, 82.0),
    "left wing back": (18.0, 78.0),
    "right wing back": (102.0, 78.0),
    "left defensive midfield": (38.0, 73.0),
    "center defensive midfield": (60.0, 72.0),
    "right defensive midfield": (82.0, 73.0),
    "left center midfield": (44.0, 67.0),
    "center midfield": (60.0, 66.0),
    "right center midfield": (76.0, 67.0),
    "left midfield": (22.0, 64.0),
    "right midfield": (98.0, 64.0),
    "left wing": (20.0, 60.0),
    "right wing": (100.0, 60.0),
    "left attacking midfield": (38.0, 61.0),
    "center attacking midfield": (60.0, 60.0),
    "right attacking midfield": (82.0, 61.0),
    "left center forward": (48.0, 56.0),
    "center forward": (60.0, 55.0),
    "right center forward": (72.0, 56.0),
    "secondary striker": (60.0, 58.0),
}


def _lineups_dir() -> Path:
    return LINEUPS_DIR


def _normalize_position_name(position_name: Any) -> str:
    text = str(position_name or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


@st.cache_data(show_spinner=False)
def _lineup_file_index(lineups_dir_str: str) -> dict[int, str]:
    base = Path(lineups_dir_str)
    if not base.exists():
        return {}
    index: dict[int, str] = {}
    for path in base.glob("*.json"):
        try:
            match_id = int(path.stem)
        except ValueError:
            continue
        index[match_id] = str(path)
    return index


@st.cache_data(show_spinner=False)
def _load_lineup_file(match_id: int, lineups_dir_str: str) -> list[dict[str, Any]] | None:
    index = _lineup_file_index(lineups_dir_str)
    file_path = index.get(int(match_id))
    if not file_path:
        return None
    try:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return None


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


def _canonical_xi_from_events(team_events: pd.DataFrame) -> pd.DataFrame:
    if team_events.empty:
        return pd.DataFrame(
            columns=["player_id", "player_name", "jersey_number", "position_id", "position_name", "sort_order", "source"]
        )

    work = team_events.copy()
    if "type_name" in work.columns:
        work = work[work["type_name"].astype("string").str.strip().str.lower() != "starting xi"]
    if "player_name" in work.columns:
        work = work[work["player_name"].astype("string").str.strip() != ""]
    if "player_id" in work.columns:
        work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce")

    for col in ("period", "minute", "second", "event_index"):
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
            pid_int = int(pid)
            if pid_int in seen_ids:
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
                "source": "events_fallback",
            }
        )
        if pd.notna(pid):
            seen_ids.add(int(pid))
        if name_key:
            seen_names.add(name_key)
        if len(out_rows) >= 11:
            break

    return pd.DataFrame(out_rows)


def _pick_team_entry(
    lineup_doc: list[dict[str, Any]],
    team_id: int | None,
    team_name: str | None,
) -> dict[str, Any] | None:
    if not lineup_doc:
        return None
    if team_id is not None:
        for entry in lineup_doc:
            if int(entry.get("team_id", -1)) == int(team_id):
                return entry
    if team_name:
        needle = team_name.strip().lower()
        for entry in lineup_doc:
            if str(entry.get("team_name") or "").strip().lower() == needle:
                return entry
    return lineup_doc[0] if len(lineup_doc) == 1 else None


def _clock_to_seconds(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 10**9
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return 10**9
    return 10**9


def _pick_primary_position(positions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not positions:
        return None
    valid = [seg for seg in positions if isinstance(seg, dict)]
    if not valid:
        return None

    starters_0000 = [
        seg
        for seg in valid
        if str(seg.get("start_reason") or "").strip().lower() == "starting xi"
        and str(seg.get("from") or "").strip() == "00:00"
    ]
    if starters_0000:
        return starters_0000[0]

    starters = [seg for seg in valid if str(seg.get("start_reason") or "").strip().lower() == "starting xi"]
    if starters:
        return sorted(starters, key=lambda seg: _clock_to_seconds(seg.get("from")))[0]

    return sorted(valid, key=lambda seg: _clock_to_seconds(seg.get("from")))[0]


def _is_starter(player: dict[str, Any]) -> bool:
    positions = [seg for seg in player.get("positions", []) if isinstance(seg, dict)]
    if not positions:
        return False
    if any(str(seg.get("start_reason") or "").strip().lower() == "starting xi" for seg in positions):
        return True
    earliest = min(positions, key=lambda seg: _clock_to_seconds(seg.get("from")))
    return str(earliest.get("from") or "").strip() == "00:00"


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


def get_starting_xi(
    fact_events: pd.DataFrame,
    match_id: int,
    team_id: int | None = None,
    team_name: str | None = None,
) -> pd.DataFrame:
    lineup_doc = _load_lineup_file(int(match_id), str(_lineups_dir()))
    if lineup_doc is not None:
        team_entry = _pick_team_entry(lineup_doc, team_id=team_id, team_name=team_name)
        if team_entry is not None:
            starters = [p for p in team_entry.get("lineup", []) if isinstance(p, dict) and _is_starter(p)]
            if not starters:
                starters = [p for p in team_entry.get("lineup", []) if isinstance(p, dict)][:11]
            rows: list[dict[str, Any]] = []
            for idx, player in enumerate(starters):
                seg = _pick_primary_position(player.get("positions", []))
                rows.append(
                    {
                        "player_id": player.get("player_id"),
                        "player_name": player.get("player_name"),
                        "jersey_number": player.get("jersey_number"),
                        "position_id": seg.get("position_id") if isinstance(seg, dict) else pd.NA,
                        "position_name": seg.get("position") if isinstance(seg, dict) else pd.NA,
                        "sort_order": idx + 1,
                        "source": "data_raw_lineups",
                    }
                )
            xi = pd.DataFrame(rows).drop_duplicates(subset=["player_id", "player_name"]).head(11).reset_index(drop=True)
            if not xi.empty:
                return xi

    team_events = _team_filter(fact_events, match_id=match_id, team_id=team_id, team_name=team_name)
    return _canonical_xi_from_events(team_events).head(11).reset_index(drop=True)


def get_formation(
    fact_events: pd.DataFrame,
    match_id: int,
    team_id: int | None = None,
    team_name: str | None = None,
) -> str | None:
    team_events = _team_filter(fact_events, match_id=match_id, team_id=team_id, team_name=team_name)
    if team_events.empty:
        return None

    starts = team_events[
        team_events.get("type_name", pd.Series("", index=team_events.index))
        .astype("string")
        .str.strip()
        .str.lower()
        == "starting xi"
    ]
    if starts.empty:
        return None
    row = starts.iloc[0]
    for col in ("tactics_formation", "formation", "team_formation", "starting_formation"):
        if col in starts.columns:
            parsed = _formation_from_text(row.get(col))
            if parsed:
                return parsed
    return None


def _coords_for_position(position_name: Any) -> tuple[float, float] | None:
    key = _normalize_position_name(position_name)
    return POSITION_COORDS_ATTACK_UP.get(key)


def _transform_half_coords(x: float, y_attack_up: float, is_home: bool) -> tuple[float, float]:
    # Home defends bottom goal and attacks upward; away is mirrored to defend top goal.
    if is_home:
        y = 100.0 - y_attack_up
    else:
        y = y_attack_up
    return x, y


def get_unmapped_position_names(
    match_id: int,
    fact_events: pd.DataFrame | None = None,
    team_id: int | None = None,
    team_name: str | None = None,
) -> list[str]:
    events = fact_events if fact_events is not None else pd.DataFrame()
    xi = get_starting_xi(events, match_id=match_id, team_id=team_id, team_name=team_name)
    if xi.empty or "position_name" not in xi.columns:
        return []
    unmapped: set[str] = set()
    for value in xi["position_name"].astype("string").fillna("").tolist():
        name = str(value).strip()
        if not name:
            continue
        if _coords_for_position(name) is None:
            unmapped.add(name)
    return sorted(unmapped)


def get_starting_positions(
    fact_events: pd.DataFrame,
    match_id: int,
    team_id: int | None = None,
    team_name: str | None = None,
    formation: str | None = None,
    half: str = "top",
    is_home: bool | None = None,
) -> list[dict[str, Any]]:
    del formation  # positions are driven by lineup JSON positions, not synthetic formation lines.
    xi = get_starting_xi(fact_events, match_id=match_id, team_id=team_id, team_name=team_name)
    if xi.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, row in xi.iterrows():
        mapped = _coords_for_position(row.get("position_name"))
        if mapped is None:
            x, y_attack_up = 60.0, 66.0
            approximate = True
        else:
            x, y_attack_up = mapped
            approximate = False

        home_flag = bool(is_home) if is_home is not None else (half.strip().lower() == "bottom")
        x, y = _transform_half_coords(float(x), float(y_attack_up), is_home=home_flag)

        jersey = row.get("jersey_number")
        jersey_text = str(int(jersey)) if pd.notna(jersey) and str(jersey).strip() else "?"
        rows.append(
            {
                "player_id": row.get("player_id"),
                "player_name": str(row.get("player_name") or "Unknown"),
                "jersey_number": row.get("jersey_number"),
                "jersey_text": jersey_text,
                "position_id": row.get("position_id"),
                "position_name": row.get("position_name"),
                "x": float(x),
                "y": float(y),
                "approximate": approximate,
            }
        )

    return rows
