# Run + Shot Schema Notes

## Running the app
- Run Streamlit from the repository root:
```bash
streamlit run app/Home.py
```
- `app` is a Python package (`app/__init__.py`), and `app/Home.py` now bootstraps repo-root imports for `app.*`.

## Shot schema changes
- `fact_shots` is normalized to IDs for shot descriptors:
  - `shot_outcome_id`
  - `shot_type_id`
  - `body_part_id`
- Readable labels come from dimensions and are mapped in the app at load time:
  - `dim_shot_outcome(shot_outcome_id -> shot_outcome_name)`
  - `dim_shot_type(shot_type_id -> shot_type_name)`
  - `dim_body_part(body_part_id -> body_part_name)`
- Redundant denormalized label columns are removed from `fact_shots`.
- New tiny dimensions exported:
  - `dim_shot_type`
  - `dim_body_part`

## Why this changed
- Keeps fact tables user-friendly without UI-side ID decoding.
- Prevents mismatched semantics across pages and KPI calculations.
