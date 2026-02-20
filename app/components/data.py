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
SAMPLE_DIR = REPO_ROOT / "data_model_sample"
LEGACY_SAMPLE_DIR = REPO_ROOT / "app" / "data" / "sample_star_schema"
LOCAL_MODEL_DIR = REPO_ROOT / "data_model"
REMOTE_CACHE_DIR = REPO_ROOT / ".cache" / "remote_star_schema"
VALID_MODES = ("sample", "remote", "local_generated")
REQUIRED_TABLES = ("dim_match", "dim_team", "dim_player", "fact_events", "fact_shots")


def _is_streamlit_cloud() -> bool:
    cloud_markers = ("STREAMLIT_SHARING_MODE", "STREAMLIT_RUNTIME", "STREAMLIT_CLOUD")
    return any(os.getenv(marker, "").strip() for marker in cloud_markers)


def _default_mode() -> str:
    env_mode = os.getenv("DATA_MODE", "").strip().lower()
    if env_mode in VALID_MODES:
        return env_mode
    if _is_streamlit_cloud() and not LOCAL_MODEL_DIR.exists():
        return "sample"
    return "sample"


def _select_mode() -> str:
    default_mode = _default_mode()
    if "data_mode" not in st.session_state:
        st.session_state["data_mode"] = default_mode

    st.sidebar.selectbox(
        "Data mode",
        options=list(VALID_MODES),
        key="data_mode",
        help=(
            "sample: curated repo sample in ./data_model_sample, remote: download packaged dataset, "
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
        if SAMPLE_DIR.exists():
            return SAMPLE_DIR
        return LEGACY_SAMPLE_DIR
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
def _read_table(path_str: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    path = Path(path_str)
    use_columns = list(columns) if columns else None
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path, columns=use_columns)
        except (KeyError, ValueError):
            if not use_columns:
                raise
            table = pd.read_parquet(path)
            existing = [c for c in use_columns if c in table.columns]
            return table[existing]
    try:
        return pd.read_csv(path, usecols=use_columns)
    except ValueError:
        if not use_columns:
            raise
        table = pd.read_csv(path)
        existing = [c for c in use_columns if c in table.columns]
        return table[existing]


def _normalize_lookup(table: pd.DataFrame, id_col: str, name_col: str) -> pd.DataFrame:
    if not {id_col, name_col}.issubset(table.columns):
        return pd.DataFrame(columns=[id_col, name_col])
    lookup = table[[id_col, name_col]].copy()
    lookup[id_col] = pd.to_numeric(lookup[id_col], errors="coerce")
    lookup = lookup.dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    lookup[id_col] = lookup[id_col].astype(int)
    lookup[name_col] = lookup[name_col].astype("string")
    return lookup


@st.cache_data(show_spinner=False)
def _load_shot_dims_cached(base_dir_str: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = Path(base_dir_str)
    dim_outcome_path = _resolve_table_file(base_dir, "dim_shot_outcome")
    dim_body_path = _resolve_table_file(base_dir, "dim_body_part")
    dim_type_path = _resolve_table_file(base_dir, "dim_shot_type")

    dim_outcome = (
        _read_table(str(dim_outcome_path), columns=("shot_outcome_id", "shot_outcome_name"))
        if dim_outcome_path is not None
        else pd.DataFrame(columns=["shot_outcome_id", "shot_outcome_name"])
    )
    dim_body = (
        _read_table(str(dim_body_path), columns=("body_part_id", "body_part_name"))
        if dim_body_path is not None
        else pd.DataFrame(columns=["body_part_id", "body_part_name"])
    )
    dim_type = (
        _read_table(str(dim_type_path), columns=("shot_type_id", "shot_type_name"))
        if dim_type_path is not None
        else pd.DataFrame(columns=["shot_type_id", "shot_type_name"])
    )
    return dim_outcome, dim_body, dim_type


def _enrich_dim_match(dim_match: pd.DataFrame, dim_team: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    enriched = dim_match.copy()

    for id_col in ("competition_id", "season_id", "home_team_id", "away_team_id"):
        if id_col in enriched.columns:
            enriched[id_col] = pd.to_numeric(enriched[id_col], errors="coerce")

    if "competition_name" not in enriched.columns and "competition_id" in enriched.columns:
        comp_path = _resolve_table_file(base_dir, "dim_competition")
        if comp_path is not None:
            comp_df = _normalize_lookup(_read_table(str(comp_path)), "competition_id", "competition_name")
            if not comp_df.empty:
                enriched = enriched.merge(comp_df, on="competition_id", how="left")

    if "season_name" not in enriched.columns and "season_id" in enriched.columns:
        season_path = _resolve_table_file(base_dir, "dim_season")
        if season_path is not None:
            season_df = _normalize_lookup(_read_table(str(season_path)), "season_id", "season_name")
            if not season_df.empty:
                enriched = enriched.merge(season_df, on="season_id", how="left")

    team_lookup = _normalize_lookup(dim_team, "team_id", "team_name")
    if not team_lookup.empty:
        if "home_team_name" not in enriched.columns and "home_team_id" in enriched.columns:
            home_lookup = team_lookup.rename(columns={"team_id": "home_team_id", "team_name": "home_team_name"})
            enriched = enriched.merge(home_lookup, on="home_team_id", how="left")
        if "away_team_name" not in enriched.columns and "away_team_id" in enriched.columns:
            away_lookup = team_lookup.rename(columns={"team_id": "away_team_id", "team_name": "away_team_name"})
            enriched = enriched.merge(away_lookup, on="away_team_id", how="left")

    return enriched


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


def _scan_relation_sql(path: Path) -> str:
    normalized = str(path).replace("\\", "/").replace("'", "''")
    if path.suffix.lower() == ".parquet":
        return f"read_parquet('{normalized}')"
    if path.suffix.lower() == ".csv":
        return f"read_csv_auto('{normalized}', header=true, sample_size=-1)"
    raise ValueError(f"Unsupported file format for {path}")


@st.cache_data(show_spinner=False)
def _get_table_columns(path_str: str) -> list[str]:
    path = Path(path_str)
    relation = _scan_relation_sql(path)
    conn = _get_duckdb_connection(str(path.parent.resolve()))
    return conn.execute(f"SELECT * FROM {relation} LIMIT 0").df().columns.tolist()


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _query_fact_rows(
    path: Path,
    match_id: int | None = None,
    team_id: int | None = None,
    player_id: int | None = None,
    selected_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = set(_get_table_columns(str(path)))
    relation = _scan_relation_sql(path)

    predicates: list[str] = []
    if match_id is not None:
        if "match_id" not in columns:
            return pd.DataFrame()
        predicates.append(f"TRY_CAST(match_id AS BIGINT) = {int(match_id)}")
    if team_id is not None and "team_id" in columns:
        predicates.append(f"TRY_CAST(team_id AS BIGINT) = {int(team_id)}")
    if player_id is not None and "player_id" in columns:
        predicates.append(f"TRY_CAST(player_id AS BIGINT) = {int(player_id)}")

    projected = [column for column in (selected_columns or []) if column in columns]
    select_clause = ", ".join(_quote_identifier(column) for column in projected) if projected else "*"
    sql = f"SELECT {select_clause} FROM {relation}"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)

    conn = _get_duckdb_connection(str(path.parent.resolve()))
    return conn.execute(sql).df()


def _concat_non_empty(frames: list[pd.DataFrame], fallback_columns: list[str]) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame(columns=fallback_columns)
    return pd.concat(valid, ignore_index=True)


def _dedupe_fact_shots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if {"match_id", "event_id"}.issubset(df.columns):
        sort_cols = [col for col in ("match_id", "event_id", "period", "minute", "second") if col in df.columns]
        ordered = df.sort_values(sort_cols) if sort_cols else df
        return ordered.drop_duplicates(subset=["match_id", "event_id"], keep="first").reset_index(drop=True)

    if "event_id" in df.columns:
        sort_cols = [col for col in ("event_id", "period", "minute", "second") if col in df.columns]
        ordered = df.sort_values(sort_cols) if sort_cols else df
        return ordered.drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)

    return df


def ensure_shot_labels(
    fact_shots: pd.DataFrame,
    dim_shot_outcome: pd.DataFrame,
    dim_body_part: pd.DataFrame,
    dim_shot_type: pd.DataFrame,
) -> pd.DataFrame:
    if fact_shots.empty:
        return fact_shots

    shots = fact_shots.copy()
    if "body_part_id" not in shots.columns and "shot_body_part_id" in shots.columns:
        shots["body_part_id"] = shots["shot_body_part_id"]

    id_columns = ("shot_outcome_id", "body_part_id", "shot_type_id")
    for col in id_columns:
        if col in shots.columns:
            shots[col] = pd.to_numeric(shots[col], errors="coerce")
            shots.loc[shots[col].notna(), col] = shots.loc[shots[col].notna(), col].astype(int)

    def _legacy_label(series: pd.Series | None) -> pd.Series:
        if series is None:
            return pd.Series(pd.NA, index=shots.index, dtype="string")
        as_text = series.astype("string").str.strip()
        numeric_like = pd.to_numeric(as_text, errors="coerce").notna()
        return as_text.where(~numeric_like)

    outcome_lookup = _normalize_lookup(dim_shot_outcome, "shot_outcome_id", "shot_outcome_name")
    body_lookup = _normalize_lookup(dim_body_part, "body_part_id", "body_part_name")
    type_lookup = _normalize_lookup(dim_shot_type, "shot_type_id", "shot_type_name")

    outcome_map = dict(zip(outcome_lookup["shot_outcome_id"], outcome_lookup["shot_outcome_name"]))
    body_map = dict(zip(body_lookup["body_part_id"], body_lookup["body_part_name"]))
    type_map = dict(zip(type_lookup["shot_type_id"], type_lookup["shot_type_name"]))

    legacy_outcome = _legacy_label(shots["shot_outcome_name"]) if "shot_outcome_name" in shots.columns else None
    if legacy_outcome is None and "shot_outcome" in shots.columns:
        legacy_outcome = _legacy_label(shots["shot_outcome"])
    elif legacy_outcome is not None and "shot_outcome" in shots.columns:
        legacy_outcome = legacy_outcome.combine_first(_legacy_label(shots["shot_outcome"]))

    legacy_body = _legacy_label(shots["body_part_name"]) if "body_part_name" in shots.columns else None
    if legacy_body is None and "body_part" in shots.columns:
        legacy_body = _legacy_label(shots["body_part"])
    elif legacy_body is not None and "body_part" in shots.columns:
        legacy_body = legacy_body.combine_first(_legacy_label(shots["body_part"]))
    if "shot_body_part_name" in shots.columns:
        legacy_body = (
            _legacy_label(shots["shot_body_part_name"])
            if legacy_body is None
            else legacy_body.combine_first(_legacy_label(shots["shot_body_part_name"]))
        )

    legacy_type = _legacy_label(shots["shot_type_name"]) if "shot_type_name" in shots.columns else None
    if legacy_type is None and "shot_type" in shots.columns:
        legacy_type = _legacy_label(shots["shot_type"])
    elif legacy_type is not None and "shot_type" in shots.columns:
        legacy_type = legacy_type.combine_first(_legacy_label(shots["shot_type"]))

    mapped_outcome = (
        shots["shot_outcome_id"].map(outcome_map).astype("string")
        if "shot_outcome_id" in shots.columns
        else pd.Series(pd.NA, index=shots.index, dtype="string")
    )
    mapped_body = (
        shots["body_part_id"].map(body_map).astype("string")
        if "body_part_id" in shots.columns
        else pd.Series(pd.NA, index=shots.index, dtype="string")
    )
    mapped_type = (
        shots["shot_type_id"].map(type_map).astype("string")
        if "shot_type_id" in shots.columns
        else pd.Series(pd.NA, index=shots.index, dtype="string")
    )

    shots["shot_outcome"] = (legacy_outcome if legacy_outcome is not None else mapped_outcome).combine_first(mapped_outcome)
    shots["body_part"] = (legacy_body if legacy_body is not None else mapped_body).combine_first(mapped_body)
    shots["shot_type"] = (legacy_type if legacy_type is not None else mapped_type).combine_first(mapped_type)
    for col in ("shot_outcome", "body_part", "shot_type"):
        shots[col] = shots[col].astype("string")

    drop_cols = [c for c in ("shot_outcome_name", "body_part_name", "shot_type_name") if c in shots.columns]
    if drop_cols:
        shots = shots.drop(columns=drop_cols)

    return shots


def _render_generation_commands() -> str:
    return (
        "python src/etl.py --input-dir data_raw --output-dir data_processed --no-append --force\n"
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
            "--no-append",
            "--force",
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


@st.cache_data(show_spinner=False)
def _load_dimensions_cached(dim_match_path: str, dim_team_path: str, dim_player_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dim_match = _read_table(dim_match_path)
    dim_team = _read_table(dim_team_path)
    dim_player = _read_table(dim_player_path)
    dim_match = _enrich_dim_match(dim_match, dim_team, Path(dim_match_path).parent)
    return dim_match, dim_team, dim_player


def _load_dimensions_from_paths(render_mode_selector: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, _, table_paths = _resolve_active_table_paths(render_mode_selector=render_mode_selector)
    return _load_dimensions_cached(
        str(table_paths["dim_match"]),
        str(table_paths["dim_team"]),
        str(table_paths["dim_player"]),
    )


def load_dimensions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _load_dimensions_from_paths(render_mode_selector=True)


def get_matches(competition_id: int | None = None, season_id: int | None = None) -> pd.DataFrame:
    dim_match, _, _ = _load_dimensions_from_paths(render_mode_selector=False)
    matches = dim_match.copy()

    if competition_id is not None and "competition_id" in matches.columns:
        matches = matches[pd.to_numeric(matches["competition_id"], errors="coerce") == int(competition_id)]
    if season_id is not None and "season_id" in matches.columns:
        matches = matches[pd.to_numeric(matches["season_id"], errors="coerce") == int(season_id)]

    if "match_label" not in matches.columns:
        home = matches["home_team_name"] if "home_team_name" in matches.columns else pd.Series("", index=matches.index)
        away = matches["away_team_name"] if "away_team_name" in matches.columns else pd.Series("", index=matches.index)
        if "match_date" in matches.columns:
            matches["match_label"] = home.astype(str) + " vs " + away.astype(str) + " (" + matches["match_date"].astype(str) + ")"
        else:
            matches["match_label"] = home.astype(str) + " vs " + away.astype(str)

    sort_cols = [col for col in ("match_date", "match_id") if col in matches.columns]
    if sort_cols:
        matches = matches.sort_values(sort_cols, ascending=False)

    return matches.reset_index(drop=True)


def get_teams_for_match(match_id: int) -> pd.DataFrame:
    dim_match, dim_team, _ = _load_dimensions_from_paths(render_mode_selector=False)
    teams: list[dict[str, object]] = []

    if {"match_id", "home_team_id", "away_team_id"}.issubset(dim_match.columns):
        dm = dim_match[pd.to_numeric(dim_match["match_id"], errors="coerce") == int(match_id)]
        if not dm.empty:
            row = dm.iloc[0]
            teams.extend(
                [
                    {"team_id": row.get("home_team_id"), "team_name": row.get("home_team_name")},
                    {"team_id": row.get("away_team_id"), "team_name": row.get("away_team_name")},
                ]
            )

    teams_df = pd.DataFrame(teams)
    if teams_df.empty:
        _, _, table_paths = _resolve_active_table_paths(render_mode_selector=False)
        candidates: list[pd.DataFrame] = []
        for table in ("fact_events", "fact_shots"):
            path = table_paths[table]
            columns = set(_get_table_columns(str(path)))
            if "match_id" not in columns or "team_id" not in columns:
                continue
            select_name = "team_name" if "team_name" in columns else "NULL::VARCHAR AS team_name"
            relation = _scan_relation_sql(path)
            conn = _get_duckdb_connection(str(path.parent.resolve()))
            q = (
                f"SELECT DISTINCT TRY_CAST(team_id AS BIGINT) AS team_id, {select_name} "
                f"FROM {relation} "
                f"WHERE TRY_CAST(match_id AS BIGINT) = {int(match_id)}"
            )
            candidates.append(conn.execute(q).df())
        teams_df = _concat_non_empty(candidates, fallback_columns=["team_id", "team_name"])

    if teams_df.empty:
        return pd.DataFrame(columns=["team_id", "team_name"])

    if "team_id" in teams_df.columns:
        teams_df["team_id"] = pd.to_numeric(teams_df["team_id"], errors="coerce")
        teams_df = teams_df.dropna(subset=["team_id"])
        teams_df["team_id"] = teams_df["team_id"].astype(int)

    if "team_name" not in teams_df.columns and {"team_id", "team_name"}.issubset(dim_team.columns):
        teams_df = teams_df.merge(
            dim_team[["team_id", "team_name"]].drop_duplicates(subset=["team_id"]),
            on="team_id",
            how="left",
        )

    if "team_name" in teams_df.columns and {"team_id", "team_name"}.issubset(dim_team.columns):
        teams_df = teams_df.merge(
            dim_team[["team_id", "team_name"]].drop_duplicates(subset=["team_id"]),
            on="team_id",
            how="left",
            suffixes=("", "_dim"),
        )
        if "team_name_dim" in teams_df.columns:
            teams_df["team_name"] = teams_df["team_name"].combine_first(teams_df["team_name_dim"])
            teams_df = teams_df.drop(columns=["team_name_dim"])

    teams_df = teams_df.dropna(subset=["team_id"]).drop_duplicates(subset=["team_id"])
    if "team_name" not in teams_df.columns:
        teams_df["team_name"] = teams_df["team_id"].astype(str)
    teams_df["team_name"] = teams_df["team_name"].fillna(teams_df["team_id"].astype(str))
    return teams_df.sort_values("team_name").reset_index(drop=True)


def get_players_for_match(match_id: int, team_id: int | None = None) -> pd.DataFrame:
    _, dim_team, dim_player = _load_dimensions_from_paths(render_mode_selector=False)
    _, _, table_paths = _resolve_active_table_paths(render_mode_selector=False)

    frames: list[pd.DataFrame] = []
    for table in ("fact_events", "fact_shots"):
        path = table_paths[table]
        columns = set(_get_table_columns(str(path)))
        if "match_id" not in columns or "player_id" not in columns:
            continue

        relation = _scan_relation_sql(path)
        select_team = "TRY_CAST(team_id AS BIGINT) AS team_id" if "team_id" in columns else "NULL::BIGINT AS team_id"
        select_name = "player_name" if "player_name" in columns else "NULL::VARCHAR AS player_name"
        where = [f"TRY_CAST(match_id AS BIGINT) = {int(match_id)}"]
        if team_id is not None and "team_id" in columns:
            where.append(f"TRY_CAST(team_id AS BIGINT) = {int(team_id)}")
        conn = _get_duckdb_connection(str(path.parent.resolve()))
        query = (
            f"SELECT DISTINCT TRY_CAST(player_id AS BIGINT) AS player_id, {select_team}, {select_name} "
            f"FROM {relation} WHERE {' AND '.join(where)}"
        )
        frames.append(conn.execute(query).df())

    if not frames:
        return pd.DataFrame(columns=["player_id", "player_name", "team_id", "team_name"])

    players = _concat_non_empty(frames, fallback_columns=["player_id", "team_id", "player_name"])
    if players.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "team_id", "team_name"])
    players = players.dropna(subset=["player_id"]).copy()
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce")
    players = players.dropna(subset=["player_id"])
    players["player_id"] = players["player_id"].astype(int)

    if "team_id" in players.columns:
        players["team_id"] = pd.to_numeric(players["team_id"], errors="coerce")
        players.loc[players["team_id"].notna(), "team_id"] = players.loc[players["team_id"].notna(), "team_id"].astype(int)

    if {"player_id", "player_name"}.issubset(dim_player.columns):
        players = players.merge(
            dim_player[["player_id", "player_name", "team_id"]].drop_duplicates(subset=["player_id"]),
            on="player_id",
            how="left",
            suffixes=("", "_dim"),
        )
        if "player_name_dim" in players.columns:
            players["player_name"] = players["player_name"].combine_first(players["player_name_dim"])
            players = players.drop(columns=["player_name_dim"])
        if "team_id_dim" in players.columns:
            players["team_id"] = players["team_id"].combine_first(players["team_id_dim"])
            players = players.drop(columns=["team_id_dim"])

    if team_id is not None and "team_id" in players.columns:
        players = players[pd.to_numeric(players["team_id"], errors="coerce") == int(team_id)]

    if {"team_id", "team_name"}.issubset(dim_team.columns):
        players = players.merge(
            dim_team[["team_id", "team_name"]].drop_duplicates(subset=["team_id"]),
            on="team_id",
            how="left",
        )
    else:
        players["team_name"] = pd.NA

    players = players.drop_duplicates(subset=["player_id"])
    if "player_name" not in players.columns:
        players["player_name"] = players["player_id"].astype(str)
    players["player_name"] = players["player_name"].fillna(players["player_id"].astype(str))
    return players.sort_values("player_name").reset_index(drop=True)


def get_shots(match_id: int, team_id: int | None = None, player_id: int | None = None) -> pd.DataFrame:
    _, base_dir, table_paths = _resolve_active_table_paths(render_mode_selector=False)
    shot_projection = [
        "event_id",
        "match_id",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "minute",
        "second",
        "period",
        "x",
        "y",
        "location_x",
        "location_y",
        "shot_end_location_x",
        "shot_end_location_y",
        "xg",
        "shot_statsbomb_xg",
        "shot_outcome_id",
        "shot_type_id",
        "body_part_id",
        # Legacy exports may still carry labels in fact_shots.
        "shot_outcome",
        "shot_outcome_name",
        "shot_type",
        "shot_type_name",
        "body_part",
        "body_part_name",
        "shot_body_part_id",
        "shot_body_part_name",
        "under_pressure",
        "play_pattern_name",
    ]
    shots = _query_fact_rows(
        table_paths["fact_shots"],
        match_id=match_id,
        team_id=team_id,
        player_id=player_id,
        selected_columns=shot_projection,
    )
    shots = _dedupe_fact_shots(shots)
    dim_shot_outcome, dim_body_part, dim_shot_type = _load_shot_dims_cached(str(base_dir.resolve()))
    shots = ensure_shot_labels(
        shots,
        dim_shot_outcome=dim_shot_outcome,
        dim_body_part=dim_body_part,
        dim_shot_type=dim_shot_type,
    )
    return shots


def get_events(match_id: int, team_id: int | None = None, player_id: int | None = None) -> pd.DataFrame:
    _, _, table_paths = _resolve_active_table_paths(render_mode_selector=False)
    return _query_fact_rows(table_paths["fact_events"], match_id=match_id, team_id=team_id, player_id=player_id)


def get_active_data_mode() -> str:
    return _active_mode_from_state()


@st.cache_data(show_spinner=False)
def _get_lineup_events_cached(match_id: int, fact_events_path: str, data_mode: str) -> pd.DataFrame:
    del data_mode  # cache key only
    projection = [
        "event_id",
        "event_index",
        "match_id",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "position_id",
        "position_name",
        "minute",
        "second",
        "period",
        "type_name",
        "tactics_formation",
        "formation",
        "team_formation",
        "starting_formation",
        "tactics_lineup",
        "lineup",
    ]
    return _query_fact_rows(Path(fact_events_path), match_id=match_id, selected_columns=projection)


def get_lineup_events(match_id: int) -> pd.DataFrame:
    data_mode = _active_mode_from_state()
    _, _, table_paths = _resolve_active_table_paths(render_mode_selector=False)
    return _get_lineup_events_cached(match_id=int(match_id), fact_events_path=str(table_paths["fact_events"]), data_mode=data_mode)


def load_fact_events(match_id: int) -> pd.DataFrame:
    return get_events(match_id=match_id)


def load_fact_shots(match_id: int) -> pd.DataFrame:
    return get_shots(match_id=match_id)
