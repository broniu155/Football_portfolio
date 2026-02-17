# Football Portfolio Streamlit App

Multi-page football analytics app with a robust data-loading strategy for local development and Streamlit Community Cloud deployment.

## Data Modes

The app supports three modes controlled by `DATA_MODE` (and overridable from the Streamlit sidebar):

- `sample` (default): uses committed files in `app/data/sample_star_schema/`
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

## Sample Dataset

Committed sample tables live in:

- `app/data/sample_star_schema/dim_match.csv`
- `app/data/sample_star_schema/dim_team.csv`
- `app/data/sample_star_schema/dim_player.csv`
- `app/data/sample_star_schema/fact_events.csv`
- `app/data/sample_star_schema/fact_shots.csv`

To regenerate sample data:

```bash
python scripts/make_sample_star_schema.py --format csv
python scripts/make_sample_star_schema.py --format parquet
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
