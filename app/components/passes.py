from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from app.components.data import get_events
except (ModuleNotFoundError, KeyError):
    from components.data import get_events

PASS_HEIGHT_CANDIDATES = ["pass_height_name", "pass_height", "pass.height.name", "pass_height__name"]
PASS_HEIGHT_ID_CANDIDATES = ["pass_height_id", "pass.height.id"]
PASS_CROSS_CANDIDATES = ["pass_cross", "pass.cross", "pass_cross_flag", "cross"]
PASS_OUTCOME_NAME_CANDIDATES = ["pass_outcome_name", "pass_outcome", "pass.outcome.name"]
PASS_OUTCOME_ID_CANDIDATES = ["pass_outcome_id", "pass.outcome.id"]


def get_filtered_events(
    match_id: int,
    team_id: int | None = None,
    player_id: int | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = events.copy() if events is not None else get_events(match_id=int(match_id))
    if work.empty:
        return work
    if "match_id" in work.columns:
        work = work[pd.to_numeric(work["match_id"], errors="coerce") == int(match_id)]
    if team_id is not None and "team_id" in work.columns:
        work = work[pd.to_numeric(work["team_id"], errors="coerce") == int(team_id)]
    if player_id is not None and "player_id" in work.columns:
        work = work[pd.to_numeric(work["player_id"], errors="coerce") == int(player_id)]
    return work


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for col in names:
        if col in df.columns:
            return col
    return None


def resolve_pass_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "height_name_col": _first_col(df, PASS_HEIGHT_CANDIDATES),
        "height_id_col": _first_col(df, PASS_HEIGHT_ID_CANDIDATES),
        "cross_col": _first_col(df, PASS_CROSS_CANDIDATES),
        "outcome_name_col": _first_col(df, PASS_OUTCOME_NAME_CANDIDATES),
        "outcome_id_col": _first_col(df, PASS_OUTCOME_ID_CANDIDATES),
    }


def _is_completed(pass_df: pd.DataFrame, columns: dict[str, str | None]) -> pd.Series:
    outcome_name_col = columns.get("outcome_name_col")
    if outcome_name_col and outcome_name_col in pass_df.columns:
        outcome = pass_df[outcome_name_col].astype("string").str.strip().str.lower()
        return outcome.isna() | (outcome == "") | (outcome == "none")
    return pd.Series(True, index=pass_df.index)


def _normalize_height(pass_df: pd.DataFrame, columns: dict[str, str | None]) -> pd.Series:
    height_col = columns.get("height_name_col")
    if not height_col or height_col not in pass_df.columns:
        return pd.Series("unknown", index=pass_df.index, dtype="string")
    normalized = pass_df[height_col].astype("string").str.strip().str.lower()
    return normalized.fillna("unknown").replace("", "unknown")


def _cross_mask(pass_df: pd.DataFrame, columns: dict[str, str | None]) -> pd.Series:
    cross_col = columns.get("cross_col")
    if not cross_col or cross_col not in pass_df.columns:
        return pd.Series(False, index=pass_df.index)
    col = pass_df[cross_col]
    if str(col.dtype).lower() in {"bool", "boolean"}:
        return col.fillna(False).astype(bool)
    text = col.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(col, errors="coerce")
    return text.isin({"true", "t", "1", "yes", "y"}) | numeric.eq(1)


def _pass_longitudinal_cols(pass_df: pd.DataFrame) -> tuple[str | None, str | None]:
    start_long = _first_col(pass_df, ["location_x", "x", "location_y", "y"])
    end_long = _first_col(pass_df, ["pass_end_location_x", "pass_end_x", "end_x", "pass_end_location_y", "pass_end_y", "end_y"])
    return start_long, end_long


def _final_third_mask(pass_df: pd.DataFrame, is_home: bool) -> pd.Series:
    _, end_long_col = _pass_longitudinal_cols(pass_df)
    if end_long_col is None:
        return pd.Series(False, index=pass_df.index)
    end_long = pd.to_numeric(pass_df[end_long_col], errors="coerce")
    if end_long.dropna().empty:
        return pd.Series(False, index=pass_df.index)

    pitch_len = 120.0 if float(end_long.max()) > 101.0 else 100.0
    high_threshold = (2.0 / 3.0) * pitch_len
    low_threshold = (1.0 / 3.0) * pitch_len
    return end_long >= high_threshold if is_home else end_long <= low_threshold


def get_pass_events(
    match_id: int,
    team_id: int | None = None,
    player_id: int | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = get_filtered_events(match_id=match_id, team_id=team_id, player_id=player_id, events=events)
    if base.empty or "type_name" not in base.columns:
        return base.iloc[0:0].copy()
    pass_df = base[base["type_name"].astype("string").str.strip().str.lower() == "pass"].copy()
    if pass_df.empty:
        return pass_df
    colmap = resolve_pass_columns(pass_df)
    pass_df["is_completed"] = _is_completed(pass_df, colmap)
    pass_df["pass_height_norm"] = _normalize_height(pass_df, colmap)
    pass_df["is_cross"] = _cross_mask(pass_df, colmap)
    return pass_df


def filter_passes(
    pass_df: pd.DataFrame,
    pass_height_filter: str = "All",
    cross_filter: str = "All",
    final_third_only: bool = False,
    is_home: bool = True,
) -> pd.DataFrame:
    work = pass_df.copy()
    if work.empty:
        return work

    colmap = resolve_pass_columns(work)
    if "pass_height_norm" not in work.columns:
        work["pass_height_norm"] = _normalize_height(work, colmap)
    if "is_cross" not in work.columns:
        work["is_cross"] = _cross_mask(work, colmap)

    target_height = {"Ground": "ground pass", "Low": "low pass", "High": "high pass"}.get(pass_height_filter)
    if target_height is not None:
        work = work[work["pass_height_norm"] == target_height]

    if cross_filter == "Crosses Only":
        work = work[work["is_cross"]]
    elif cross_filter == "Exclude Crosses":
        work = work[~work["is_cross"]]

    if final_third_only:
        work = work[_final_third_mask(work, is_home=is_home)]
    return work


def _missing_breakdown_fields(pass_df: pd.DataFrame) -> bool:
    colmap = resolve_pass_columns(pass_df)
    return colmap.get("height_name_col") is None or colmap.get("cross_col") is None


def _na_stats() -> dict[str, float | None]:
    return {
        "total_passes": 0,
        "completed_passes": 0,
        "completion_pct": 0.0,
        "progressive_passes": 0,
        "final_third_passes": 0,
        "ground_passes": None,
        "low_passes": None,
        "high_passes": None,
        "crosses": None,
        "cross_completion_pct": None,
    }


def compute_pass_stats(pass_df: pd.DataFrame, is_home: bool) -> dict[str, float | None]:
    if pass_df.empty:
        return _na_stats()

    work = pass_df.copy()
    colmap = resolve_pass_columns(work)
    if "pass_height_norm" not in work.columns:
        work["pass_height_norm"] = _normalize_height(work, colmap)
    if "is_cross" not in work.columns:
        work["is_cross"] = _cross_mask(work, colmap)

    start_long_col, end_long_col = _pass_longitudinal_cols(work)
    progressive = 0
    if start_long_col and end_long_col:
        start_long = pd.to_numeric(work[start_long_col], errors="coerce")
        end_long = pd.to_numeric(work[end_long_col], errors="coerce")
        delta = end_long - start_long if is_home else start_long - end_long
        progressive = int((delta >= 15.0).fillna(False).sum())

    total = int(len(work))
    completed = int(work["is_completed"].sum()) if "is_completed" in work.columns else 0
    completion_pct = (completed / total * 100.0) if total else 0.0
    final_third = int(_final_third_mask(work, is_home=is_home).sum())

    if _missing_breakdown_fields(work):
        stats = _na_stats()
        stats["total_passes"] = total
        stats["completed_passes"] = completed
        stats["completion_pct"] = completion_pct
        stats["progressive_passes"] = progressive
        stats["final_third_passes"] = final_third
        return stats

    ground = int((work["pass_height_norm"] == "ground pass").sum())
    low = int((work["pass_height_norm"] == "low pass").sum())
    high = int((work["pass_height_norm"] == "high pass").sum())
    crosses = int(work["is_cross"].sum())
    cross_completed = int((work["is_cross"] & work["is_completed"]).sum())
    cross_completion_pct = (cross_completed / crosses * 100.0) if crosses else 0.0

    return {
        "total_passes": total,
        "completed_passes": completed,
        "completion_pct": completion_pct,
        "progressive_passes": progressive,
        "final_third_passes": final_third,
        "ground_passes": ground,
        "low_passes": low,
        "high_passes": high,
        "crosses": crosses,
        "cross_completion_pct": cross_completion_pct,
    }


def _pitch_shapes(line_color: str = "#6c8f78") -> list[dict[str, Any]]:
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=line_color, width=2)),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=59, y0=39, x1=61, y1=41, fillcolor=line_color, line=dict(color=line_color)),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=11, y0=39, x1=13, y1=41, fillcolor=line_color, line=dict(color=line_color)),
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=107, y0=39, x1=109, y1=41, fillcolor=line_color, line=dict(color=line_color)),
    ]
    for shape in shapes:
        shape["layer"] = "below"
    return shapes


def _line_trace(df: pd.DataFrame, sx_col: str, sy_col: str, ex_col: str, ey_col: str, color: str, name: str) -> go.Scattergl:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for row in df.itertuples(index=False):
        sx = getattr(row, sx_col)
        sy = getattr(row, sy_col)
        ex = getattr(row, ex_col)
        ey = getattr(row, ey_col)
        if pd.isna(sx) or pd.isna(sy) or pd.isna(ex) or pd.isna(ey):
            continue
        x_values.extend([float(sx), float(ex), None])
        y_values.extend([float(sy), float(ey), None])
    return go.Scattergl(x=x_values, y=y_values, mode="lines", line=dict(width=1.4, color=color), name=name, opacity=0.62)


def _endpoint_markers(df: pd.DataFrame, x_col: str, y_col: str, color: str, size: float, opacity: float) -> go.Scattergl:
    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce")
    keep = x_values.notna() & y_values.notna()
    return go.Scattergl(
        x=x_values[keep],
        y=y_values[keep],
        mode="markers",
        marker=dict(size=size, color=color, opacity=opacity, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    )


def render_pass_map(pass_df: pd.DataFrame, title: str, max_lines: int = 1100) -> go.Figure:
    fig = go.Figure()
    if pass_df.empty:
        fig.update_layout(
            title=title,
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111a2b",
            font=dict(color="#e7edf7"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500,
            shapes=_pitch_shapes(),
            margin=dict(l=10, r=10, t=55, b=10),
        )
        return fig

    start_x_col = _first_col(pass_df, ["location_x", "x"])
    start_y_col = _first_col(pass_df, ["location_y", "y"])
    end_x_col = _first_col(pass_df, ["pass_end_location_x", "pass_end_x", "end_x"])
    end_y_col = _first_col(pass_df, ["pass_end_location_y", "pass_end_y", "end_y"])
    if not all([start_x_col, start_y_col, end_x_col, end_y_col]):
        return fig

    work = pass_df.copy()
    for col in [start_x_col, start_y_col, end_x_col, end_y_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[start_x_col, start_y_col, end_x_col, end_y_col])
    if work.empty:
        return fig
    if len(work) > max_lines:
        work = work.sample(n=max_lines, random_state=42)

    completed = work[work["is_completed"]]
    unsuccessful = work[~work["is_completed"]]
    if not completed.empty:
        fig.add_trace(_line_trace(completed, start_x_col, start_y_col, end_x_col, end_y_col, "#42d392", "Completed"))
        fig.add_trace(_endpoint_markers(completed, start_x_col, start_y_col, "#7de8b8", size=4.2, opacity=0.45))
        fig.add_trace(_endpoint_markers(completed, end_x_col, end_y_col, "#42d392", size=8.4, opacity=0.85))
    if not unsuccessful.empty:
        fig.add_trace(_line_trace(unsuccessful, start_x_col, start_y_col, end_x_col, end_y_col, "#f59e0b", "Unsuccessful"))
        fig.add_trace(_endpoint_markers(unsuccessful, start_x_col, start_y_col, "#f8c46d", size=4.2, opacity=0.45))
        fig.add_trace(_endpoint_markers(unsuccessful, end_x_col, end_y_col, "#f59e0b", size=8.4, opacity=0.85))

    fig.update_layout(
        title=title,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111a2b",
        font=dict(color="#e7edf7"),
        margin=dict(l=10, r=10, t=55, b=10),
        height=500,
        shapes=_pitch_shapes(),
        legend=dict(orientation="h", yanchor="top", y=1.04, xanchor="left", x=0),
    )
    fig.update_xaxes(range=[-2, 122], visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-2, 82], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    return fig


def _fmt_metric(value: float | None, is_pct: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}%" if is_pct else str(int(value))


def render_pass_comparison_panel(home_stats: dict[str, float | None], away_stats: dict[str, float | None], home_name: str, away_name: str) -> None:
    rows = [
        ("Total Passes", home_stats["total_passes"], away_stats["total_passes"], False),
        ("Completed Passes", home_stats["completed_passes"], away_stats["completed_passes"], False),
        ("Pass Completion %", home_stats["completion_pct"], away_stats["completion_pct"], True),
        ("Progressive Passes", home_stats["progressive_passes"], away_stats["progressive_passes"], False),
        ("Passes into Final Third", home_stats["final_third_passes"], away_stats["final_third_passes"], False),
        ("Ground Passes", home_stats["ground_passes"], away_stats["ground_passes"], False),
        ("Low Passes", home_stats["low_passes"], away_stats["low_passes"], False),
        ("High Passes", home_stats["high_passes"], away_stats["high_passes"], False),
        ("Crosses", home_stats["crosses"], away_stats["crosses"], False),
        ("Cross Completion %", home_stats["cross_completion_pct"], away_stats["cross_completion_pct"], True),
    ]
    html = ['<div class="match-stats-panel">']
    html.append(f'<div class="match-stats-context">{home_name} vs {away_name}</div>')
    for label, hv, av, is_pct in rows:
        html.append(
            "<div class='match-stats-row'><div class='match-stats-values'>"
            f"<div class='home'>{_fmt_metric(hv, is_pct=is_pct)}</div>"
            f"<div class='label'>{label}</div>"
            f"<div class='away'>{_fmt_metric(av, is_pct=is_pct)}</div>"
            "</div></div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _team_subset(pass_df: pd.DataFrame, team_id: int, selected_team_id: int | None, selected_player_id: int | None) -> pd.DataFrame:
    work = pass_df.copy()
    if work.empty:
        return work
    if selected_team_id is not None and int(selected_team_id) != int(team_id):
        return work.iloc[0:0].copy()
    if "team_id" in work.columns:
        work = work[pd.to_numeric(work["team_id"], errors="coerce") == int(team_id)]
    if selected_player_id is not None and "player_id" in work.columns:
        work = work[pd.to_numeric(work["player_id"], errors="coerce") == int(selected_player_id)]
    return work


def _debug_enabled() -> bool:
    session_debug = bool(st.session_state.get("debug")) or bool(st.session_state.get("debug_passes"))
    query_debug = False
    try:
        qv = st.query_params.get("debug", "")
        query_debug = str(qv).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        query_debug = False
    return session_debug or query_debug


def _render_debug_expander(events_for_match: pd.DataFrame, pass_df: pd.DataFrame) -> None:
    subset = [c for c in events_for_match.columns if any(k in c.lower() for k in ["pass_height", "pass_cross", "cross", "pass_outcome"])]
    with st.expander("Pass Debug", expanded=False):
        st.write("Candidate columns:", subset)
        colmap = resolve_pass_columns(events_for_match)
        st.write("Resolved mapping:", colmap)
        if colmap.get("height_name_col") and colmap["height_name_col"] in pass_df.columns:
            vals = (
                pass_df[colmap["height_name_col"]]
                .astype("string")
                .dropna()
                .str.strip()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            st.write("Height unique values:", vals)
        cross_col = colmap.get("cross_col")
        if cross_col and cross_col in pass_df.columns:
            mask = _cross_mask(pass_df, colmap)
            st.write("Cross distribution:", {"true": int(mask.sum()), "false": int((~mask).sum())})


def render_passes_section(
    match_id: int,
    home_team_id: int | None,
    away_team_id: int | None,
    home_team_name: str,
    away_team_name: str,
    selected_team_id: int | None,
    selected_player_id: int | None,
    events: pd.DataFrame,
) -> None:
    events_for_match = get_filtered_events(match_id=match_id, events=events)
    pass_df = get_pass_events(match_id=match_id, team_id=None, player_id=None, events=events_for_match)
    if home_team_id is None or away_team_id is None:
        st.info("Home/Away team metadata is unavailable for this match.")
        return

    colmap = resolve_pass_columns(pass_df if not pass_df.empty else events_for_match)
    missing_breakdown = colmap.get("height_name_col") is None or colmap.get("cross_col") is None
    if missing_breakdown:
        st.warning(
            "Pass height/cross fields are missing from fact_events export. "
            "Re-run ETL + export_star_schema to include them."
        )

    home_passes = _team_subset(pass_df, team_id=int(home_team_id), selected_team_id=selected_team_id, selected_player_id=selected_player_id)
    away_passes = _team_subset(pass_df, team_id=int(away_team_id), selected_team_id=selected_team_id, selected_player_id=selected_player_id)

    f1, f2 = st.columns(2)
    with f1:
        height_filter = st.selectbox("Pass Height Filter", options=["All", "Ground", "Low", "High"], index=0, key="passes_height_filter")
    with f2:
        cross_filter = st.selectbox(
            "Cross Filter",
            options=["All", "Crosses Only", "Exclude Crosses"],
            index=0,
            key="passes_cross_filter",
        )

    filtered_home = filter_passes(home_passes, pass_height_filter=height_filter, cross_filter=cross_filter, final_third_only=False, is_home=True)
    filtered_away = filter_passes(away_passes, pass_height_filter=height_filter, cross_filter=cross_filter, final_third_only=False, is_home=False)

    st.markdown('<div class="section-title">Pass Comparison Stats</div>', unsafe_allow_html=True)
    render_pass_comparison_panel(
        compute_pass_stats(filtered_home, is_home=True),
        compute_pass_stats(filtered_away, is_home=False),
        home_team_name,
        away_team_name,
    )

    st.markdown('<div class="section-title">Home vs Away Pass Maps</div>', unsafe_allow_html=True)
    c_home, c_away = st.columns(2)
    with c_home:
        st.plotly_chart(render_pass_map(filtered_home, title="Home Passes"), use_container_width=True)
    with c_away:
        st.plotly_chart(render_pass_map(filtered_away, title="Away Passes"), use_container_width=True)

    st.markdown('<div class="section-title">Passes into Final Third</div>', unsafe_allow_html=True)
    ft_home, ft_away = st.columns(2)
    with ft_home:
        final_home = filter_passes(home_passes, pass_height_filter=height_filter, cross_filter=cross_filter, final_third_only=True, is_home=True)
        st.plotly_chart(render_pass_map(final_home, title="Home Final Third Passes"), use_container_width=True)
    with ft_away:
        final_away = filter_passes(away_passes, pass_height_filter=height_filter, cross_filter=cross_filter, final_third_only=True, is_home=False)
        st.plotly_chart(render_pass_map(final_away, title="Away Final Third Passes"), use_container_width=True)

    if _debug_enabled():
        _render_debug_expander(events_for_match, pass_df)
