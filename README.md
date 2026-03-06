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

## Attack Channel Zoning

`attack_channel` uses StatsBomb pitch coordinates on a 120x80 surface (`x` in `[0,120]`, `y` in `[0,80]`).
Lane bins are defined on `y` without orientation flipping in raw data:

- `Left`: `0 <= y < 80/3`
- `Centre`: `80/3 <= y < 160/3`
- `Right`: `160/3 <= y <= 80`

For event exports, coordinate source precedence is:
`event.location` (`location_x`,`location_y`) first, then `pass.end_location`, `carry.end_location`, and `shot.end_location` (first two values if shot has `[x,y,z]`).
Debug columns `channel_source`, `channel_reason`, `x_used`, and `y_used` are included with derived export fields.

## Dribble Counting (StatsBomb Open Data Events v4.0.0)

Dribbles are counted from `fact_events` rows where `type_name == "Dribble"` (or canonical dribble `type_id` where present).

- Attempts: every dribble event.
- Completed: `dribble_outcome_name == "Complete"`.
- Incomplete: `dribble_outcome_name == "Incomplete"`.
- Missing/blank outcome: treated as `Unknown` and not counted as complete.

This follows the StatsBomb Events v4.0.0 dribble structure (`type=Dribble` with dribble outcome in the dribble object).

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

## Live Pitch Replay (Beta)

Match Report -> Analysis View -> Transitions now includes a collapsed **Live Pitch Replay (Beta)** block.

- Base mode uses event coordinates from `fact_events` (ball, actor, and optional event path).
- When available, it overlays StatsBomb 360 freeze-frame players from `fact_three_sixty_freeze_frames`.
- Visible-area polygons are optional and shown only when discoverable (for example from local `data_processed/three_sixty_visible_area`).

Coverage note in UI:

- `Events-only`: replay animates event actor/ball without 360 context.
- `Events+360`: replay includes freeze-frame overlays.

Known limitations:

- This is not continuous tracking data; frames are event-tied snapshots.
- Animation interpolates between event start/end points and does not reconstruct full off-ball movement.

## Set Piece Tactical View (Phase 1)

Match Report -> Analysis View -> Set Pieces now includes a tactical view focused on corners and free kicks.

Phase 1 includes:

- rule-based extraction of corner/free-kick events
- single-event tactical pitch delivery plot
- aggregate pattern summary (type/side/zone/subtype)
- simple summary metrics (`set pieces`, `linked shots`, `linked goals`, `short routines %`)

Validation:

```bash
python -m unittest tests.test_set_pieces -v
```

Manual verification:

1. Run `streamlit run app/pages/1_Match_Report.py`
2. Select a match with corner/free-kick events.
3. Open `Set Pieces` tab.
4. Confirm:
   - `Set Piece Type` filter updates the view,
   - single event delivery line is shown on pitch,
   - aggregate pattern table + target-zone chart render,
   - summary metrics change with filter.
