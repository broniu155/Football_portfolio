# Data Quality Report

## Run Mode
- Append mode: True

## Processor File Summary
- `events`: found=3465, skipped_unchanged=3465, processed=0, parse_failures=0
- `competitions`: found=1, skipped_unchanged=1, processed=0, parse_failures=0
- `matches`: found=75, skipped_unchanged=75, processed=0, parse_failures=0
- `lineups`: found=3464, skipped_unchanged=3464, processed=0, parse_failures=0
- `three-sixty`: found=326, skipped_unchanged=1, processed=325, parse_failures=6

## Output Row Counts (This Run)
- `events`: 0 rows
- `competitions`: 0 rows
- `matches`: 0 rows
- `lineups_players`: 0 rows
- `three_sixty`: 1,024,538 rows
- `three_sixty_freeze_frames`: 15,538,303 rows
- `three_sixty_visible_area`: 1,024,538 rows
- `teams`: 22 rows
- `players`: 38 rows

## Required Column Checks
- `events`: PASS
- `competitions`: PASS
- `matches`: PASS
- `lineups_players`: PASS
- `three_sixty`: PASS
- `three_sixty_freeze_frames`: PASS
- `three_sixty_visible_area`: PASS
- `teams`: PASS
- `players`: PASS

## Key ID Null Rates
- `three_sixty`: match_id=0.00%, event_uuid=0.00% (sampled rows: 1,024,538)
- `three_sixty_freeze_frames`: match_id=0.00%, event_uuid=0.00%, player_id=100.00% (sampled rows: 15,538,303)
- `three_sixty_visible_area`: match_id=0.00%, event_uuid=0.00% (sampled rows: 1,024,538)
- `teams`: team_id=0.00% (sampled rows: 22)
- `players`: player_id=0.00%, team_id=0.00% (sampled rows: 38)
