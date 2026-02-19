# Data Quality Report

## Run Mode
- Append mode: False

## Processor File Summary
- `events`: found=3465, skipped_unchanged=0, processed=3465, parse_failures=0
- `competitions`: found=1, skipped_unchanged=0, processed=1, parse_failures=0
- `matches`: found=75, skipped_unchanged=0, processed=75, parse_failures=0
- `lineups`: found=3464, skipped_unchanged=0, processed=3464, parse_failures=0
- `three-sixty`: found=326, skipped_unchanged=0, processed=326, parse_failures=6

## Output Row Counts (This Run)
- `events`: 12,188,951 rows
- `competitions`: 75 rows
- `matches`: 3,464 rows
- `lineups_players`: 131,901 rows
- `three_sixty`: 1,027,908 rows
- `three_sixty_freeze_frames`: 15,584,040 rows
- `three_sixty_visible_area`: 1,027,908 rows
- `teams`: 314 rows
- `players`: 10,805 rows

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
- `events`: match_id=0.00%, team_id=0.00%, player_id=0.43% (sampled rows: 12,188,951)
- `competitions`: competition_id=0.00%, season_id=0.00% (sampled rows: 75)
- `matches`: match_id=0.00%, competition_id=0.00%, season_id=0.00%, home_team_id=0.00%, away_team_id=0.00% (sampled rows: 3,464)
- `lineups_players`: match_id=0.00%, team_id=0.00%, player_id=0.00% (sampled rows: 131,901)
- `three_sixty`: match_id=0.00%, event_uuid=0.00% (sampled rows: 1,027,908)
- `three_sixty_freeze_frames`: match_id=0.00%, event_uuid=0.00%, player_id=100.00% (sampled rows: 15,584,040)
- `three_sixty_visible_area`: match_id=0.00%, event_uuid=0.00% (sampled rows: 1,027,908)
- `teams`: team_id=0.00% (sampled rows: 314)
- `players`: player_id=0.00%, team_id=0.00% (sampled rows: 10,805)
