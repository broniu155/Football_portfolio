from __future__ import annotations

import streamlit as st

ANALYSIS_VIEWS = ("Shots", "Passes", "Duels / Recoveries", "More")


def render_analysis_nav(current_view: str = "Shots") -> str:
    if "analysis_view" not in st.session_state:
        st.session_state["analysis_view"] = current_view if current_view in ANALYSIS_VIEWS else "Shots"
    elif st.session_state["analysis_view"] not in ANALYSIS_VIEWS:
        st.session_state["analysis_view"] = "Shots"

    st.markdown('<div class="analysis-nav-title">Analysis View</div>', unsafe_allow_html=True)
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Analysis View",
            options=list(ANALYSIS_VIEWS),
            key="analysis_view",
            label_visibility="collapsed",
        )
    else:
        selected = st.radio(
            "Analysis View",
            options=list(ANALYSIS_VIEWS),
            key="analysis_view",
            horizontal=True,
            label_visibility="collapsed",
        )

    return str(selected or st.session_state.get("analysis_view") or "Shots")
