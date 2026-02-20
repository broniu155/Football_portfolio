# Code Audit Report

Date: 2026-02-20

## Scope and Safety Rules
- Audited `app/`, `src/`, `scripts/`, `tests/`, top-level docs, and page discovery paths.
- Applied conservative deletion rules: remove only when no repo references and no dynamic-loading risk.
- Preserved Streamlit entry points and all ETL/export/sample workflows.

## Baseline Verification (Before Cleanup)
1. Compile check:
   - Command: `.\.venv\Scripts\python.exe -m compileall .`
   - Result: PASS
2. Smoke imports:
   - Command: `.\.venv\Scripts\python.exe -c "import importlib;mods=['app.Home','app.components.data','app.components.filters'];[importlib.import_module(m) for m in mods];print('smoke_imports_ok')"`
   - Result: PASS (`smoke_imports_ok`)
3. Existing tests:
   - Command: `.\.venv\Scripts\python.exe -m pytest -q`
   - Result: NOT RUN (environment missing `pytest`)
   - Existing tests are present (`tests/test_normalize.py`), so no new smoke test file was added.

## Audit Findings

### Removed (Provably Unused)
1. `app/components/match_summary.py`
   - Evidence:
     - No imports/references found via repo search (`rg -n "render_match_summary\(|match_summary"`).
     - Not part of Streamlit page discovery (`app/pages/` unaffected).
     - No side-effect usage (module only defines helper functions; no startup registration).
   - Action:
     - Deleted file.
   - Behavior impact:
     - None expected; module was unreachable from current app/CLI flow.

### Candidates Kept (Uncertain or Still Referenced)
1. `src/tactical_metrics.py`
   - Referenced in `notebooks/01_match_report.ipynb`; kept.
2. `src/viz.py`
   - Referenced in `notebooks/01_match_report.ipynb`; kept.
3. `app/components/data.py` wrappers `load_fact_events`, `load_fact_shots`
   - No current in-repo callers, but likely compatibility/public helpers; kept.
4. `app/components/filters.py` wrapper `sidebar_filters`
   - No current in-repo callers, but likely compatibility/public helper; kept.

## Files Changed
1. `app/components/match_summary.py`
   - Deleted as dead module (no references).
2. `reports/code_audit.md`
   - Added audit evidence, decisions, and verification logs.

## Post-Cleanup Verification
1. Compile check:
   - Command: `.\.venv\Scripts\python.exe -m compileall .`
   - Result: PASS
2. Smoke imports:
   - Command: `.\.venv\Scripts\python.exe -c "import importlib;mods=['app.Home','app.components.data','app.components.filters'];[importlib.import_module(m) for m in mods];print('smoke_imports_ok')"`
   - Result: PASS (`smoke_imports_ok`)
3. Existing tests:
   - Command: `.\.venv\Scripts\python.exe -m pytest -q`
   - Result: NOT RUN (environment missing `pytest`)

## Functional-Change Check
- No changes were made to page scripts, filtering logic, ETL pipeline logic, export logic, schema generation, or UI behavior paths.
- Streamlit entry points and `app/pages/` discovery remain unchanged.
- Data loading modes (`sample`, `remote`, `local_generated`) remain unchanged.
