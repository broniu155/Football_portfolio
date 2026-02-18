from pathlib import Path

import duckdb
import streamlit as st

from app.components.ui import setup_page

setup_page(page_title="Data Model", page_icon="🗂️")

st.title("Star Schema (Data Model)")
st.caption("Preview tables from `data_model/` without loading entire datasets into memory.")

data_model = Path("data_model")
if not data_model.exists():
    st.error("Missing `data_model/` folder.")
    st.stop()

tables = sorted([p for p in data_model.glob("*.parquet")] + [p for p in data_model.glob("*.csv")])
if not tables:
    st.warning("No CSV/Parquet files found in `data_model/`.")
    st.stop()

choice = st.selectbox("Choose a table", [p.name for p in tables])
table_path = data_model / choice

conn = duckdb.connect(database=":memory:")
if table_path.suffix.lower() == ".parquet":
    relation = f"read_parquet('{str(table_path).replace('\\', '/')}')"
else:
    relation = f"read_csv_auto('{str(table_path).replace('\\', '/')}', header=true, sample_size=-1)"

row_count = conn.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
preview_df = conn.execute(f"SELECT * FROM {relation} LIMIT 200").df()

st.write(f"Rows: {row_count:,} • Columns: {len(preview_df.columns):,}")
st.markdown('<div class="section-title">Preview (Top 200)</div>', unsafe_allow_html=True)
st.dataframe(preview_df, use_container_width=True)
