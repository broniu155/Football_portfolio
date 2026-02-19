from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate star-schema data quality for football analytics app.")
    p.add_argument("--model-dir", type=Path, default=Path("data_model"))
    p.add_argument("--match-id", type=int, default=None)
    p.add_argument("--team-id", type=int, default=None)
    p.add_argument("--player-id", type=int, default=None)
    return p.parse_args()


def table_relation(model_dir: Path, table: str) -> str:
    parquet_path = model_dir / f"{table}.parquet"
    csv_path = model_dir / f"{table}.csv"
    if parquet_path.exists():
        return f"read_parquet('{str(parquet_path).replace('\\', '/')}')"
    if csv_path.exists():
        return f"read_csv_auto('{str(csv_path).replace('\\', '/')}', header=true, sample_size=-1)"
    raise FileNotFoundError(f"Missing table: {table}.parquet or {table}.csv")


def one(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0])


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir
    if not model_dir.exists():
        print(f"[FAIL] Model directory not found: {model_dir}")
        return 1

    con = duckdb.connect()
    required = ["fact_shots", "dim_match", "dim_team", "dim_player"]

    relations: dict[str, str] = {}
    try:
        for table in required:
            relations[table] = table_relation(model_dir, table)
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        return 1

    dim_outcome_rel: str | None = None
    try:
        dim_outcome_rel = table_relation(model_dir, "dim_shot_outcome")
    except FileNotFoundError:
        dim_outcome_rel = None

    failed = False

    dup_match_event = one(
        con,
        f"""
        SELECT COALESCE(SUM(n - 1), 0)
        FROM (
          SELECT CAST(match_id AS VARCHAR) AS match_id, CAST(event_id AS VARCHAR) AS event_id, COUNT(*) AS n
          FROM {relations['fact_shots']}
          GROUP BY 1, 2
          HAVING COUNT(*) > 1
        )
        """,
    )
    if dup_match_event > 0:
        print(f"[FAIL] fact_shots duplicate (match_id,event_id) extra rows: {dup_match_event}")
        failed = True
    else:
        print("[PASS] fact_shots has unique (match_id,event_id).")

    # If fact_shots has outcome_name, it should be populated; otherwise validate via dim mapping if available.
    fact_shots_cols = {
        r[0].lower()
        for r in con.execute(f"SELECT * FROM {relations['fact_shots']} LIMIT 0").description
    }
    has_outcome_name = "shot_outcome_name" in fact_shots_cols
    has_outcome_id = "shot_outcome_id" in fact_shots_cols
    has_legacy_outcome = "shot_outcome" in fact_shots_cols

    if has_legacy_outcome and has_outcome_name:
        redundant_outcome = one(
            con,
            f"""
            SELECT COUNT(*)
            FROM {relations['fact_shots']}
            WHERE lower(trim(COALESCE(shot_outcome, ''))) = lower(trim(COALESCE(shot_outcome_name, '')))
              AND NULLIF(trim(COALESCE(shot_outcome_name, '')), '') IS NOT NULL
            """,
        )
        if redundant_outcome > 0:
            print(
                "[FAIL] Redundant shot outcome columns detected: "
                f"{redundant_outcome} rows have identical shot_outcome and shot_outcome_name."
            )
            failed = True

    missing_labels = 0
    if has_outcome_id:
        if has_outcome_name and dim_outcome_rel is not None:
            missing_labels = one(
                con,
                f"""
                SELECT COUNT(*)
                FROM {relations['fact_shots']} f
                LEFT JOIN {dim_outcome_rel} d
                  ON TRY_CAST(f.shot_outcome_id AS BIGINT) = TRY_CAST(d.shot_outcome_id AS BIGINT)
                WHERE f.shot_outcome_id IS NOT NULL
                  AND NULLIF(TRIM(COALESCE(f.shot_outcome_name, d.shot_outcome_name, '')), '') IS NULL
                """,
            )
        elif dim_outcome_rel is not None:
            missing_labels = one(
                con,
                f"""
                SELECT COUNT(*)
                FROM {relations['fact_shots']} f
                LEFT JOIN {dim_outcome_rel} d
                  ON TRY_CAST(f.shot_outcome_id AS BIGINT) = TRY_CAST(d.shot_outcome_id AS BIGINT)
                WHERE f.shot_outcome_id IS NOT NULL
                  AND NULLIF(TRIM(COALESCE(d.shot_outcome_name, '')), '') IS NULL
                """,
            )
        else:
            print("[WARN] dim_shot_outcome not found; skipping outcome label coverage check.")

    if missing_labels > 0:
        print(f"[FAIL] fact_shots rows with missing outcome labels: {missing_labels}")
        failed = True
    elif has_outcome_id and dim_outcome_rel is not None:
        print("[PASS] Shot outcome labels are available for mapped outcome IDs.")

    fk_checks = [
        ("match_id", "dim_match", "match_id"),
        ("team_id", "dim_team", "team_id"),
        ("player_id", "dim_player", "player_id"),
    ]
    for fact_key, dim_table, dim_key in fk_checks:
        if fact_key not in fact_shots_cols:
            continue
        missing_fk = one(
            con,
            f"""
            SELECT COUNT(*)
            FROM {relations['fact_shots']} f
            LEFT JOIN {relations[dim_table]} d
              ON TRY_CAST(f.{fact_key} AS BIGINT) = TRY_CAST(d.{dim_key} AS BIGINT)
            WHERE f.{fact_key} IS NOT NULL
              AND d.{dim_key} IS NULL
            """,
        )
        if missing_fk > 0:
            print(f"[FAIL] fact_shots.{fact_key} missing in {dim_table}.{dim_key}: {missing_fk}")
            failed = True
        else:
            print(f"[PASS] Referential integrity for fact_shots.{fact_key} -> {dim_table}.{dim_key}")

    label_pairs = [("shot_type_id", "shot_type_name"), ("body_part_id", "body_part_name")]
    for id_col, name_col in label_pairs:
        if id_col not in fact_shots_cols:
            continue
        if name_col not in fact_shots_cols:
            print(f"[FAIL] fact_shots has {id_col} but missing {name_col}.")
            failed = True
            continue
        missing_pair_labels = one(
            con,
            f"""
            SELECT COUNT(*)
            FROM {relations['fact_shots']}
            WHERE {id_col} IS NOT NULL
              AND NULLIF(TRIM(COALESCE({name_col}, '')), '') IS NULL
            """,
        )
        if missing_pair_labels > 0:
            print(f"[FAIL] fact_shots rows missing {name_col} for populated {id_col}: {missing_pair_labels}")
            failed = True
        else:
            print(f"[PASS] {id_col} has matching {name_col} labels.")

    if has_outcome_id and dim_outcome_rel is not None:
        where_parts = ["1=1"]
        if args.match_id is not None:
            where_parts.append(f"TRY_CAST(f.match_id AS BIGINT) = {int(args.match_id)}")
        if args.team_id is not None:
            where_parts.append(f"TRY_CAST(f.team_id AS BIGINT) = {int(args.team_id)}")
        if args.player_id is not None:
            where_parts.append(f"TRY_CAST(f.player_id AS BIGINT) = {int(args.player_id)}")
        where_sql = " AND ".join(where_parts)
        outcome_name_expr = "f.shot_outcome_name" if has_outcome_name else "NULL"

        goal_name_count = one(
            con,
            f"""
            SELECT COUNT(*)
            FROM {relations['fact_shots']} f
            LEFT JOIN {dim_outcome_rel} d
              ON TRY_CAST(f.shot_outcome_id AS BIGINT) = TRY_CAST(d.shot_outcome_id AS BIGINT)
            WHERE {where_sql}
              AND lower(trim(COALESCE({outcome_name_expr}, d.shot_outcome_name, ''))) = 'goal'
            """,
        )
        goal_ids = con.execute(
            f"""
            SELECT DISTINCT TRY_CAST(shot_outcome_id AS BIGINT)
            FROM {dim_outcome_rel}
            WHERE lower(trim(COALESCE(shot_outcome_name, ''))) = 'goal'
              AND TRY_CAST(shot_outcome_id AS BIGINT) IS NOT NULL
            """
        ).fetchall()
        if goal_ids:
            in_list = ",".join(str(int(row[0])) for row in goal_ids)
            goal_id_count = one(
                con,
                f"""
                SELECT COUNT(*)
                FROM {relations['fact_shots']} f
                WHERE {where_sql}
                  AND TRY_CAST(f.shot_outcome_id AS BIGINT) IN ({in_list})
                """,
            )
            if goal_name_count != goal_id_count:
                print(
                    "[FAIL] Goal semantic mismatch in filtered context: "
                    f"name-based={goal_name_count}, id-based={goal_id_count}"
                )
                failed = True
            else:
                print(
                    "[PASS] Goal semantic consistency in filtered context: "
                    f"{goal_name_count} goals."
                )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
