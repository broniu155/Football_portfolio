# Run + Shot Schema Notes

## Running the app
- Run Streamlit from the repository root:
```bash
streamlit run app/Home.py
```
- `app` is a Python package (`app/__init__.py`), and `app/Home.py` now bootstraps repo-root imports for `app.*`.

## Shot schema changes
- `fact_shots` uses canonical outcome fields:
  - `shot_outcome_id`
  - `shot_outcome_name`
- Redundant legacy outcome text (`shot_outcome`) is not required for new exports.
- `fact_shots` now includes readable shot labels:
  - `shot_type_id` + `shot_type_name`
  - `body_part_id` + `body_part_name`
- New tiny dimensions exported:
  - `dim_shot_type`
  - `dim_body_part`

## Why this changed
- Keeps fact tables user-friendly without UI-side ID decoding.
- Prevents mismatched semantics across pages and KPI calculations.
