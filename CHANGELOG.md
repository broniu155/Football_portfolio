# Changelog

## 2026-03-06
- Added Phase 1 `Set Piece Tactical View` in `Match Report > Set Pieces`:
  - Extracts corners and free kicks from match events.
  - Adds rule-based tactical enrichment (side, subtype, target zone, short routine flag).
  - Adds single-event tactical pitch delivery view.
  - Adds aggregate pattern view and simple summary metrics.
  - Adds follow-up linkage flags (`linked_shot`, `linked_goal`) using next-actions/time-window heuristics.
- Added unit tests for set-piece extraction and rule classifications.
- Added dedicated non-UI data enrichment module `app/components/set_piece_data.py` for set-piece extraction and classification.
- Added `scripts/validate_set_piece_data.py` to print schema and sample rows for a selected match.
- Added unit tests for the pure data module in `tests/test_set_piece_data.py`.
- Added sanity-check aggregation helper (`compute_set_piece_sanity_checks`) for classification QA metrics.
- Added documented rule definitions in `docs/set_piece_metric_definitions.md`.
- Expanded Match Report `Set Pieces` UI with dedicated filters and three tactical tabs (`Single Event`, `Pattern View`, `Summary`).
