from __future__ import annotations

from pathlib import Path

import streamlit as st


def setup_page(page_title: str, page_icon: str) -> None:
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    css_path = Path(__file__).resolve().parents[1] / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
