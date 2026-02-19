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


def _normalize_lookup(table: pd.DataFrame, id_col: str, name_col: str) -> pd.DataFrame:
    if not {id_col, name_col}.issubset(table.columns):
        return pd.DataFrame(columns=[id_col, name_col])
    lookup = table[[id_col, name_col]].copy()
    lookup[id_col] = pd.to_numeric(lookup[id_col], errors="coerce")
    lookup = lookup.dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    lookup[id_col] = lookup[id_col].astype(int)
    lookup[name_col] = lookup[name_col].astype("string")
    return lookup


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


def _query_fact_rows(
    path: Path,
    match_id: int | None = None,
    team_id: int | None = None,
    player_id: int | None = None,
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

    sql = f"SELECT * FROM {relation}"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)

    conn = _get_duckdb_connection(str(path.parent.resolve()))
    return conn.execute(sql).df()


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


def _enrich_shot_outcome_name(shots: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    if shots.empty:
        return shots

    enriched = shots.copy()
    if "shot_outcome_name" not in enriched.columns:
        enriched["shot_outcome_name"] = pd.NA

    if "shot_outcome" in enriched.columns:
        # Preserve existing labels if available in older files.
        enriched["shot_outcome_name"] = enriched["shot_outcome_name"].combine_first(enriched["shot_outcome"])

    if "shot_outcome_id" in enriched.columns:
        dim_path = _resolve_table_file(base_dir, "dim_shot_outcome")
        if dim_path is not None:
            dim_outcome = _read_table(str(dim_path))
            if {"shot_outcome_id", "shot_outcome_name"}.issubset(dim_outcome.columns):
                lookup = dim_outcome[["shot_outcome_id", "shot_outcome_name"]].copy()
                lookup["shot_outcome_id"] = pd.to_numeric(lookup["shot_outcome_id"], errors="coerce")
                lookup = lookup.dropna(subset=["shot_outcome_id"]).drop_duplicates(subset=["shot_outcome_id"])
                lookup["shot_outcome_id"] = lookup["shot_outcome_id"].astype(int)
                lookup["shot_outcome_name"] = lookup["shot_outcome_name"].astype("string")

                enriched["shot_outcome_id"] = pd.to_numeric(enriched["shot_outcome_id"], errors="coerce")
                enriched = enriched.merge(lookup, on="shot_outcome_id", how="left", suffixes=("", "_dim"))
                if "shot_outcome_name_dim" in enriched.columns:
                    enriched["shot_outcome_name"] = enriched["shot_outcome_name"].combine_first(
                        enriched["shot_outcome_name_dim"]
                    )
                    enriched = enriched.drop(columns=["shot_outcome_name_dim"])

    return enriched


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
        if candidates:
            teams_df = pd.concat(candidates, ignore_index=True)

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

    players = pd.concat(frames, ignore_index=True)
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
    shots = _query_fact_rows(table_paths["fact_shots"], match_id=match_id, team_id=team_id, player_id=player_id)
    shots = _dedupe_fact_shots(shots)
    shots = _enrich_shot_outcome_name(shots, base_dir=base_dir)
    return shots


def get_events(match_id: int, team_id: int | None = None, player_id: int | None = None) -> pd.DataFrame:
    _, _, table_paths = _resolve_active_table_paths(render_mode_selector=False)
    return _query_fact_rows(table_paths["fact_events"], match_id=match_id, team_id=team_id, player_id=player_id)


def load_fact_events(match_id: int) -> pd.DataFrame:
    return get_events(match_id=match_id)


def load_fact_shots(match_id: int) -> pd.DataFrame:
    return get_shots(match_id=match_id)
