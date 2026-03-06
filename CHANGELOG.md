# Changelog

## 2026-03-06
- Added Phase 1 `Set Piece Tactical View` in `Match Report > Set Pieces`:
  - Extracts corners and free kicks from match events.
  - Adds rule-based tactical enrichment (side, subtype, target zone, short routine flag).
  - Adds single-event tactical pitch delivery view.
  - Adds aggregate pattern view and simple summary metrics.
  - Adds follow-up linkage flags (`linked_shot`, `linked_goal`) using next-actions/time-window heuristics.
- Added unit tests for set-piece extraction and rule classifications.
