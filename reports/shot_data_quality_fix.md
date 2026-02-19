# Shot Data Quality + Semantics Fix

## Root Cause
- `data_processed/events.csv` had duplicated rows from append/incremental ETL runs.
- `fact_shots` was built from that source without a uniqueness guard, so KPIs were inflated.
- `fact_shots` stored `shot_outcome_id` but not `shot_outcome_name`, so UI views often showed IDs/codes.
- Goal KPIs in pages relied on direct string equality to `"Goal"`, which failed when only IDs were present.

## Fix Implemented
- `src/export_star_schema.py`
  - `fact_shots` now includes `shot_outcome_name`.
  - Deterministic dedupe applied during export using `(match_id, event_id)` for both `fact_events` and `fact_shots`.
  - Duplicate skips are logged as warnings.
- `app/components/data.py`
  - `get_shots()` now defensively dedupes by `(match_id, event_id)` (fallback to `event_id`).
  - Backward compatibility: if old models lack `shot_outcome_name`, labels are enriched via `dim_shot_outcome`.
  - Local generation command now uses `--no-append --force` to prevent duplicate buildup.
- `app/components/model_views.py`
  - Shot outcome display now prioritizes label fields.
  - Added `is_goal` semantic flag with fallback to StatsBomb goal outcome id (`97`).
- Streamlit pages updated to compute Goals from `is_goal` (fallback to case-insensitive label check).

## Validation
1. Rebuild model:
```bash
python src/etl.py --input-dir data_raw --output-dir data_processed --no-append --force
python src/export_star_schema.py --input-dir data_processed --output-dir data_model --format parquet
```
2. Run validation checks:
```bash
python src/validate_data_model.py --model-dir data_model
```
3. Optional targeted check (Boniface example):
```bash
python src/validate_data_model.py --model-dir data_model --match-id <match_id> --player-id <player_id>
```
