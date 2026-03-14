# Set Piece Tactical Definitions (Rule-Based)

Scope: corners and free kicks from `fact_events` where `play_pattern_name` is `From Corner` or `From Free Kick`, and event type is `Pass` or `Shot`.

## Core fields

- `set_piece_type`:
  - `Corner` if `play_pattern_name == From Corner`
  - `Free Kick` if `play_pattern_name == From Free Kick`
- `start_x`, `start_y`: event start location (`location_x`, `location_y`)
- `end_x`, `end_y`:
  - pass end (`pass_end_location_x`, `pass_end_location_y`) when available
  - otherwise shot end (`shot_end_location_x`, `shot_end_location_y`)
  - fallback to start location if end location is missing

## Tactical classifications

- `short_set_piece`:
  - `True` when `(end_x - start_x) < 10` (StatsBomb 120x80 pitch units)
  - otherwise `False`

- `side`:
  - For corners:
    - `Left` if `start_y < 40`
    - `Right` if `start_y >= 40`
  - For free kicks:
    - `Left` if `start_y < 26.67`
    - `Centre` if `26.67 <= start_y <= 53.33`
    - `Right` if `start_y > 53.33`

- `free_kick_type`:
  - `Direct` if event type is `Shot`
  - `Crossed` if `pass_cross` is truthy (`true/1/yes/y/t`)
  - `Short` if not direct/crossed and `(end_x - start_x) < 10`
  - `Indirect` otherwise

- `target_zone`:
  - `Recycled/Short` if `end_x < 90`
  - `Six-yard central` if `end_x >= 114` and `30 <= end_y <= 50`
  - `Near-post` if `end_x >= 108` and `end_y < 30`
  - `Far-post` if `end_x >= 108` and `end_y > 50`
  - `Penalty area` if `end_x >= 100` and `18 <= end_y <= 62`
  - `Edge/Other` otherwise

- `subtype`:
  - if `short_set_piece == True`: `Short routine`
  - Corners:
    - `Post delivery` for `Near-post`/`Far-post`
    - `Box delivery` for `Six-yard central`/`Penalty area`
    - `Recycled` otherwise
  - Free kicks:
    - `Direct shot` for `free_kick_type == Direct`
    - `Crossed delivery` for `free_kick_type == Crossed`
    - `Indirect routine` otherwise

- `outcome`:
  - For passes: `Complete` if pass outcome is blank/null, else pass outcome label
  - For shots: shot outcome label, else `Unknown`

## Linked follow-up outcomes

- `linked_shot`:
  - `True` if same-team shot occurs within:
    - next `N` events (`event_index` window), and
    - next `S` seconds (`minute/second` window)
  - defaults: `N=5`, `S=15`

- `linked_goal`:
  - `True` if a linked shot has outcome `Goal`

## Sanity-check metrics

- Counts by `set_piece_type`
- Distribution by `side`
- Distribution by `target_zone`
- Totals for `linked_shot` and `linked_goal`

## Defensive corner exits

Scope: `Clearance` events where `play_pattern_name == From Corner`.

- `clearance_x`, `clearance_y`: the clearance event location.
- `exit_x`, `exit_y`:
  - estimated from the first subsequent actionable event with a valid location
  - search window defaults: next `6` actions and next `12` seconds
  - non-ball actions such as `Pressure` are ignored for this estimate

- `exit_lane`:
  - `Left` if `exit_y < 26.67`
  - `Centre` if `26.67 <= exit_y <= 53.33`
  - `Right` if `exit_y > 53.33`
  - `Unknown` if no valid follow-up location is found

- `first_ball_winner`:
  - `Defending team` if the first actionable follow-up event is by the clearance team
  - `Attacking team` if the first actionable follow-up event is by the corner-taking team
  - `Unknown` if no qualifying follow-up event is found

- Lane summary outputs:
  - total clearances per lane
  - `share_pct` of clearances by lane
  - defending-team first-ball counts and percentages by lane
