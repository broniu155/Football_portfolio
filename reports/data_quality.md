# Data Quality Report

## Pipeline Summary
- Event files processed: 3465
- Event rows written: 12188951
- Unique matches: 3465

## Required Columns Check
- Status: PASS
- Missing columns: none
- Null-rate stats based on all 12,188,951 rows

## Stable ID Coverage
- `match_id` null rate: 0.00%
- `team_id` null rate: 0.00%
- `player_id` null rate: 0.43%

## Coordinate Coverage
- `location_x` null rate: 0.75%
- `location_y` null rate: 0.75%
- `pass_end_location_x` null rate: 72.21%
- `pass_end_location_y` null rate: 72.21%

## Notes
- One row is produced per event record from source JSON arrays.
- `match_id` is derived from source filenames.
- Nested structures are flattened with underscore-separated columns.
