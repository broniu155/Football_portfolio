# Data Quality Report

## Pipeline Summary
- Event files processed: 25
- Event rows written: 95861
- Unique matches: 25

## Required Columns Check
- Status: PASS
- Missing columns: none

## Stable ID Coverage
- `match_id` null rate: 0.00%
- `team_id` null rate: 0.00%
- `player_id` null rate: 0.34%

## Coordinate Coverage
- `location_x` null rate: 0.59%
- `location_y` null rate: 0.59%
- `pass_end_location_x` null rate: 71.41%
- `pass_end_location_y` null rate: 71.41%

## Notes
- One row is produced per event record from source JSON arrays.
- `match_id` is derived from source filenames.
- Nested structures are flattened with underscore-separated columns.
