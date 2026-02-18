from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "app" / "data" / "sample_star_schema"
LOCAL_MODEL_DIR = REPO_ROOT / "data_model"
REMOTE_CACHE_DIR = REPO_ROOT / ".cache" / "remote_star_schema"
VALID_MODES = ("sample", "remote", "local_generated")
REQUIRED_TABLES = ("dim_match", "dim_team", "dim_player", "fact_events", "fact_shots")


def _default_mode() -> str:
    env_mode = os.getenv("DATA_MODE", "sample").strip().lower()
    return env_mode if env_mode in VALID_MODES else "sample"


def _select_mode() -> str:
    default_mode = _default_mode()
    if "data_mode" not in st.session_state:
        st.session_state["data_mode"] = default_mode

    st.sidebar.selectbox(
        "Data mode",
        options=list(VALID_MODES),
        key="data_mode",
        help=(
            "sample: built-in demo data, remote: download packaged dataset, "
            "local_generated: use ./data_model generated from local raw JSON."
        ),
    )
    return st.session_state["data_mode"]


def _active_mode_from_state() -> str:
    mode = str(st.session_state.get("data_mode", _default_mode())).strip().lower()
    return mode if mode in VALID_MODES else _default_mode()


@st.cache_resource(show_spinner=False)
def _prepare_remote_data(data_url: str) -> str:
    REMOTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    marker = REMOTE_CACHE_DIR / ".ready"
    if marker.exists():
        return str(REMOTE_CACHE_DIR)

    archive_path = REMOTE_CACHE_DIR / "dataset.zip"
    urllib.request.urlretrieve(data_url, archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(REMOTE_CACHE_DIR)
    marker.write_text("ok", encoding="utf-8")
    return str(REMOTE_CACHE_DIR)


def _resolve_data_dir(active_mode: str) -> Path:
    if active_mode == "sample":
        return SAMPLE_DIR
    if active_mode == "local_generated":
        return LOCAL_MODEL_DIR

    data_url = os.getenv("DATA_URL", "").strip()
    if not data_url:
        raise RuntimeError(
            "DATA_URL is not set for remote mode. Set DATA_URL to a public zip containing star-schema files."
        )
    return Path(_prepare_remote_data(data_url))


def _resolve_table_file(base_dir: Path, table_name: str) -> Path | None:
    parquet_path = base_dir / f"{table_name}.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = base_dir / f"{table_name}.csv"
    if csv_path.exists():
        return csv_path
    return None


def _validate_required_tables(base_dir: Path) -> tuple[list[str], dict[str, Path]]:
    table_paths: dict[str, Path] = {}
    missing: list[str] = []

    for table in REQUIRED_TABLES:
        resolved = _resolve_table_file(base_dir, table)
        if resolved is None:
            missing.append(f"{table}.parquet or {table}.csv")
        else:
            table_paths[table] = resolved

    return missing, table_paths


@st.cache_data(show_spinner=False)
def _read_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def _get_duckdb_connection(cache_key: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(database=":memory:")
    conn.execute("PRAGMA threads=4")
    return conn


def _fact_scan_sql(path: Path) -> tuple[str, list[object]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "read_parquet(?)", [str(path)]
    if suffix == ".csv":
        return "read_csv_auto(?, header=true, sample_size=-1)", [str(path)]
    raise ValueError(f"Unsupported file format for {path}")


def _load_fact_by_match(path: Path, match_id: int) -> pd.DataFrame:
    scan_sql, scan_params = _fact_scan_sql(path)
    conn = _get_duckdb_connection(str(path.parent.resolve()))
    query = (
        f"SELECT * "
        f"FROM {scan_sql} "
        f"WHERE TRY_CAST(match_id AS BIGINT) = ?"
    )
    return conn.execute(query, [*scan_params, int(match_id)]).df()


def _render_generation_commands() -> str:
    return (
        "python src/etl.py --input-dir data_raw --output-dir data_processed\n"
        "python src/export_star_schema.py --input-dir data_processed --output-dir data_model --format parquet"
    )


def _run_local_generation() -> None:
    commands = [
        [
            sys.executable,
            str(REPO_ROOT / "src" / "etl.py"),
            "--input-dir",
            str(REPO_ROOT / "data_raw"),
            "--output-dir",
            str(REPO_ROOT / "data_processed"),
        ],
        [
            sys.executable,
            str(REPO_ROOT / "src" / "export_star_schema.py"),
            "--input-dir",
            str(REPO_ROOT / "data_processed"),
            "--output-dir",
            str(REPO_ROOT / "data_model"),
            "--format",
            "parquet",
        ],
    ]

    with st.spinner("Generating local data model..."):
        for command in commands:
            rendered = " ".join(command)
            st.code(f"$ {rendered}", language="bash")
            result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
            output = "\n".join([result.stdout.strip(), result.stderr.strip()]).strip()
            if output:
                st.code(output, language="text")
            if result.returncode != 0:
                st.error(f"Command failed with exit code {result.returncode}: {rendered}")
                st.stop()

    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Local data model generated successfully. Reloading data...")


def _render_missing_data_error(active_mode: str, resolved_dir: Path, missing_files: list[str]) -> None:
    missing_text = "\n".join(f"- {item}" for item in missing_files) if missing_files else "- all required files"
    st.error(
        "\n".join(
            [
                "Required star-schema tables are missing.",
                f"Active data mode: {active_mode}",
                f"Resolved data directory: {resolved_dir}",
                "",
                "Missing files:",
                missing_text,
                "",
                "Local fix commands:",
                _render_generation_commands(),
            ]
        )
    )


def _resolve_active_table_paths(render_mode_selector: bool = False) -> tuple[str, Path, dict[str, Path]]:
    active_mode = _select_mode() if render_mode_selector else _active_mode_from_state()
    try:
        resolved_dir = _resolve_data_dir(active_mode)
    except Exception as exc:
        st.error(
            "\n".join(
                [
                    "Unable to prepare data directory.",
                    f"Active data mode: {active_mode}",
                    f"Error: {exc}",
                ]
            )
        )
        st.stop()

    missing_files: list[str] = []
    if not resolved_dir.exists():
        missing_files = [f"{table}.parquet or {table}.csv" for table in REQUIRED_TABLES]
        table_paths: dict[str, Path] = {}
    else:
        missing_files, table_paths = _validate_required_tables(resolved_dir)

    if missing_files:
        _render_missing_data_error(active_mode, resolved_dir, missing_files)
        if active_mode == "local_generated":
            st.info("Generate local data model from data_raw and data_processed.")
            if st.button("Generate Local Data Model", type="primary"):
                _run_local_generation()
                st.rerun()
        st.stop()

    return active_mode, resolved_dir, table_paths


def load_dimensions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, _, table_paths = _resolve_active_table_paths(render_mode_selector=True)
    dim_match = _read_table(str(table_paths["dim_match"]))
    dim_team = _read_table(str(table_paths["dim_team"]))
    dim_player = _read_table(str(table_paths["dim_player"]))
    return dim_match, dim_team, dim_player


def load_fact_events(match_id: int) -> pd.DataFrame:
    _, _, table_paths = _resolve_active_table_paths(render_mode_selector=False)
    return _load_fact_by_match(table_paths["fact_events"], match_id)


def load_fact_shots(match_id: int) -> pd.DataFrame:
    _, _, table_paths = _resolve_active_table_paths(render_mode_selector=False)
    return _load_fact_by_match(table_paths["fact_shots"], match_id)
