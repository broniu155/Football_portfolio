import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Data Model", page_icon="🗂️", layout="wide")

st.title("🗂️ Star Schema (Data Model)")

st.markdown("This page previews the exported star schema tables in `data_model/`.")

data_model = Path("data_model")
if not data_model.exists():
    st.error("Missing data_model/ folder.")
    st.stop()

csvs = sorted([p for p in data_model.glob("*.csv")])
choice = st.selectbox("Choose a table", [p.name for p in csvs])

df = pd.read_csv(data_model / choice)
st.write(f"Rows: {len(df):,} • Columns: {len(df.columns):,}")
st.dataframe(df.head(200), use_container_width=True)
