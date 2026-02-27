from __future__ import annotations

import html
from typing import Iterable

import pandas as pd
import streamlit as st


def _format_value(value: float | int | str | None, is_pct: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        text = value.strip()
        return text if text else "N/A"
    if isinstance(value, float):
        if pd.isna(value):
            return "N/A"
        if is_pct:
            return f"{value:.1f}%"
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}"
    if is_pct:
        return f"{float(value):.1f}%"
    return str(int(value))


def render_comparison_panel(
    rows: Iterable[tuple[str, float | int | str | None, float | int | str | None, bool]],
    home_name: str,
    away_name: str,
) -> None:
    content = ['<div class="match-stats-panel">']
    content.append(f'<div class="match-stats-context">{html.escape(home_name)} vs {html.escape(away_name)}</div>')
    for label, home_val, away_val, is_pct in rows:
        content.append(
            "<div class='match-stats-row'><div class='match-stats-values'>"
            f"<div class='home'>{html.escape(_format_value(home_val, is_pct=is_pct))}</div>"
            f"<div class='label'>{html.escape(label)}</div>"
            f"<div class='away'>{html.escape(_format_value(away_val, is_pct=is_pct))}</div>"
            "</div></div>"
        )
    content.append("</div>")
    st.markdown("".join(content), unsafe_allow_html=True)
