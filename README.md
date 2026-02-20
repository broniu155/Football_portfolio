# Football Portfolio Streamlit App

Multi-page football analytics app with a robust data-loading strategy for local development and Streamlit Community Cloud deployment.

## Data Modes

The app supports three modes controlled by `DATA_MODE` (and overridable from the Streamlit sidebar):

- `sample` (default): uses committed curated sample files in `data_model_sample/`
- `remote`: downloads a packaged dataset from `DATA_URL` on first run
- `local_generated`: loads from `data_model/` generated from local `data_raw/`

Required tables for pages:

- `dim_match`
- `dim_team`
- `dim_player`
- `fact_events`
- `fact_shots`

The loader reads `*.parquet` first, then falls back to `*.csv`.

## Quickstart (Local)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/pages/1_Match_Report.py
```

Default mode is `sample`, so app starts without private raw data.

## Build Local Data Model From `data_raw/`

Set mode:

```bash
# Windows PowerShell
$env:DATA_MODE="local_generated"
# macOS/Linux
export DATA_MODE=local_generated
```

Then either:

- use the in-app **Generate Local Data Model** button when files are missing, or
- run commands manually:

```bash
python src/etl.py --input-dir data_raw --output-dir data_processed
python src/export_star_schema.py --input-dir data_processed --output-dir data_model --format parquet
```

## Export Format

`src/export_star_schema.py` now supports:

```bash
python src/export_star_schema.py --input-dir data_processed --output-dir data_model --format parquet
python src/export_star_schema.py --input-dir data_processed --output-dir data_model --format csv
```

Default format is `parquet`.

Shot schema note: IDs live in `fact_shots` (`shot_outcome_id`, `body_part_id`, `shot_type_id`),
while readable labels come from dimensions and are mapped in the app at load time.

## Sample Dataset

Committed sample data lives in `data_model_sample/` and is safe to commit.
Raw StatsBomb JSON and full local model outputs are not committed.

To regenerate sample data locally (Bundesliga 2023/2024, default 10 matches):

```bash
python src/build_sample_data_model.py --n-matches 10
```

## Streamlit Community Cloud

Recommended defaults:

- `DATA_MODE=sample` (or leave unset)
- Do not run ETL/export during app startup

Optional remote mode:

- `DATA_MODE=remote`
- `DATA_URL=<public zip url containing star-schema files>`

If required tables are missing, the app shows a professional error with:

- active mode
- resolved data path
- missing files
- exact local fix commands

## Starting XI & Formation (Match Report)

Match Report now includes a **Starting XI & Formation** section for both teams in the selected match.

- Primary source: `fact_events` rows where `type_name == "Starting XI"`.
- If exported tactics/lineup fields are present, formation/lineup are parsed from those fields.
- If lineup details are missing (common in compact exports), the app falls back to:
  - XI inferred from the earliest 11 distinct player appearances per team
  - formation inferred from available position labels
  - approximate on-pitch placement from formation shape

Fallback behavior is non-fatal:

- if lineup data is incomplete or missing, the page shows warnings instead of failing.

To regenerate model files locally:

```bash
python src/etl.py --input-dir data_raw --output-dir data_processed --no-append --force
python src/export_star_schema.py --input-dir data_processed --output-dir data_model --format parquet
```
