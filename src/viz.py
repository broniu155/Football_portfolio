from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch


FIGURE_DIR = Path("reports/figures")
COLOR_BG = "#F8FAFC"
COLOR_LINE = "#1F2937"
COLOR_PRIMARY = "#1D4ED8"
COLOR_SECONDARY = "#0F766E"
COLOR_ACCENT = "#DC2626"


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _setup_pitch(
    title: str,
    figsize: tuple[int, int] = (12, 8),
    facecolor: str = COLOR_BG,
) -> tuple[plt.Figure, plt.Axes]:
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=facecolor,
        line_color=COLOR_LINE,
        linewidth=1.2,
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(facecolor)
    ax.set_title(title, fontsize=14, color=COLOR_LINE, pad=12)
    return fig, ax


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def plot_team_heatmap(
    events: pd.DataFrame,
    team_id: int,
    output_path: str | Path = FIGURE_DIR / "team_heatmap.png",
    x_col: str = "location_x",
    y_col: str = "location_y",
    team_col: str = "team_id",
    bins: tuple[int, int] = (12, 8),
) -> Path:
    """Plot a team touch-density heatmap and save as PNG.

    Football logic:
    - Uses event start locations for one team.
    - Darker zones indicate more frequent ball actions.
    """
    team_mask = _to_numeric(events[team_col]) == team_id
    x = _to_numeric(events.loc[team_mask, x_col])
    y = _to_numeric(events.loc[team_mask, y_col])

    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    fig, ax = _setup_pitch(f"Team Heatmap | Team {team_id}")
    pitch = Pitch(pitch_type="statsbomb")
    heat = pitch.bin_statistic(x, y, statistic="count", bins=bins)
    pitch.heatmap(heat, ax=ax, cmap="Blues", edgecolors=COLOR_BG, alpha=0.9)

    return _save_figure(fig, output_path)


def plot_passing_network(
    passes: pd.DataFrame,
    team_id: int,
    output_path: str | Path = FIGURE_DIR / "passing_network.png",
    passer_col: str = "player_id",
    receiver_col: str = "pass_recipient_id",
    team_col: str = "team_id",
    x_col: str = "location_x",
    y_col: str = "location_y",
) -> Path:
    """Plot a simple passing network and save as PNG.

    Football logic:
    - Node size scales with player pass volume.
    - Edge width scales with pass count between player pairs.
    - Uses pass start location mean per player as node position.
    """
    team_mask = _to_numeric(passes[team_col]) == team_id
    df = passes.loc[team_mask].copy()

    required = [passer_col, receiver_col, x_col, y_col]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    if df.empty:
        fig, ax = _setup_pitch(f"Passing Network | Team {team_id}")
        ax.text(60, 40, "No passing data", ha="center", va="center", color=COLOR_LINE)
        return _save_figure(fig, output_path)

    df[x_col] = _to_numeric(df[x_col])
    df[y_col] = _to_numeric(df[y_col])

    node_pos = (
        df.groupby(passer_col, as_index=False)
        .agg(
            x=(x_col, "mean"),
            y=(y_col, "mean"),
            touches=(passer_col, "count"),
        )
        .rename(columns={passer_col: "player_id"})
    )

    edges = (
        df.groupby([passer_col, receiver_col], as_index=False)
        .size()
        .rename(columns={"size": "pass_count"})
    )

    node_lookup = node_pos.set_index("player_id")[["x", "y"]]
    edges = edges[
        edges[passer_col].isin(node_lookup.index) & edges[receiver_col].isin(node_lookup.index)
    ].copy()

    fig, ax = _setup_pitch(f"Passing Network | Team {team_id}")

    max_edge = edges["pass_count"].max() if not edges.empty else 1
    for _, edge in edges.iterrows():
        p1 = edge[passer_col]
        p2 = edge[receiver_col]
        x1, y1 = node_lookup.loc[p1, "x"], node_lookup.loc[p1, "y"]
        x2, y2 = node_lookup.loc[p2, "x"], node_lookup.loc[p2, "y"]
        width = 0.5 + 3.0 * (edge["pass_count"] / max_edge)
        ax.plot([x1, x2], [y1, y2], color=COLOR_SECONDARY, linewidth=width, alpha=0.45)

    max_touch = node_pos["touches"].max() if not node_pos.empty else 1
    sizes = 120 + 900 * (node_pos["touches"] / max_touch)
    ax.scatter(
        node_pos["x"],
        node_pos["y"],
        s=sizes,
        c=COLOR_PRIMARY,
        edgecolors=COLOR_BG,
        linewidths=1.2,
        alpha=0.95,
        zorder=3,
    )

    for _, row in node_pos.iterrows():
        ax.text(
            row["x"],
            row["y"],
            str(int(row["player_id"])),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            zorder=4,
        )

    return _save_figure(fig, output_path)


def plot_shot_map_xg(
    shots: pd.DataFrame,
    team_id: int,
    output_path: str | Path = FIGURE_DIR / "shot_map_xg.png",
    team_col: str = "team_id",
    x_col: str = "location_x",
    y_col: str = "location_y",
    xg_col: str = "shot_statsbomb_xg",
    outcome_col: str = "shot_outcome_name",
) -> Path:
    """Plot shot locations with marker size scaled by xG and save as PNG.

    Football logic:
    - Each marker is a shot event.
    - Marker area scales with expected goals (xG).
    - Goals are highlighted with accent color.
    """
    team_mask = _to_numeric(shots[team_col]) == team_id
    df = shots.loc[team_mask].copy()
    df[x_col] = _to_numeric(df[x_col])
    df[y_col] = _to_numeric(df[y_col])
    df[xg_col] = _to_numeric(df[xg_col]).fillna(0.0)
    df = df.dropna(subset=[x_col, y_col])

    fig, ax = _setup_pitch(f"Shot Map (xG) | Team {team_id}")
    if df.empty:
        ax.text(60, 40, "No shot data", ha="center", va="center", color=COLOR_LINE)
        return _save_figure(fig, output_path)

    goal_mask = df[outcome_col].fillna("").eq("Goal") if outcome_col in df.columns else pd.Series(False, index=df.index)
    sizes = 80 + df[xg_col].clip(lower=0.0) * 1400

    ax.scatter(
        df.loc[~goal_mask, x_col],
        df.loc[~goal_mask, y_col],
        s=sizes[~goal_mask],
        c=COLOR_PRIMARY,
        alpha=0.55,
        edgecolors=COLOR_LINE,
        linewidths=0.5,
        label="Shot",
    )
    ax.scatter(
        df.loc[goal_mask, x_col],
        df.loc[goal_mask, y_col],
        s=sizes[goal_mask],
        c=COLOR_ACCENT,
        alpha=0.85,
        edgecolors=COLOR_LINE,
        linewidths=0.6,
        label="Goal",
    )
    ax.legend(loc="upper left", frameon=False)

    return _save_figure(fig, output_path)


def plot_defensive_action_map(
    actions: pd.DataFrame,
    team_id: int,
    output_path: str | Path = FIGURE_DIR / "defensive_action_map.png",
    team_col: str = "team_id",
    type_col: str = "type_name",
    x_col: str = "location_x",
    y_col: str = "location_y",
) -> Path:
    """Plot defensive action locations and save as PNG.

    Football logic:
    - Plots defensive events (e.g., pressure, duel, interception) spatially.
    - Marker color reflects action type for quick tactical reading.
    """
    team_mask = _to_numeric(actions[team_col]) == team_id
    df = actions.loc[team_mask].copy()
    df[x_col] = _to_numeric(df[x_col])
    df[y_col] = _to_numeric(df[y_col])
    df = df.dropna(subset=[x_col, y_col])

    fig, ax = _setup_pitch(f"Defensive Action Map | Team {team_id}")
    if df.empty:
        ax.text(60, 40, "No defensive data", ha="center", va="center", color=COLOR_LINE)
        return _save_figure(fig, output_path)

    palette = {
        "Pressure": COLOR_PRIMARY,
        "Duel": COLOR_SECONDARY,
        "Interception": COLOR_ACCENT,
        "Block": "#7C3AED",
        "Ball Recovery": "#EA580C",
        "Clearance": "#0EA5E9",
        "Foul Committed": "#B91C1C",
    }

    for action_type, group in df.groupby(type_col):
        color = palette.get(action_type, COLOR_LINE)
        ax.scatter(
            group[x_col],
            group[y_col],
            s=38,
            c=color,
            alpha=0.78,
            edgecolors=COLOR_BG,
            linewidths=0.4,
            label=action_type,
        )

    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    return _save_figure(fig, output_path)

