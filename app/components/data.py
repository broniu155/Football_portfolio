import pandas as pd
import streamlit as st
from pathlib import Path

DATA_MODEL_DIR = Path("data_model")

@st.cache_data(show_spinner=False)
def load_star_schema():
    """
    Cache heavy reads so navigation between pages is fast.
    For large files, consider converting CSV -> Parquet offline and reading Parquet.
    """
    dim_match = pd.read_csv(DATA_MODEL_DIR / "dim_match.csv")
    dim_team = pd.read_csv(DATA_MODEL_DIR / "dim_team.csv")
    dim_player = pd.read_csv(DATA_MODEL_DIR / "dim_player.csv")
    fact_events = pd.read_csv(DATA_MODEL_DIR / "fact_events.csv")
    fact_shots = pd.read_csv(DATA_MODEL_DIR / "fact_shots.csv")
    return dim_match, dim_team, dim_player, fact_events, fact_shots
