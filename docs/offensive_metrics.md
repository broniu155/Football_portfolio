# Offensive Metrics (Match Report)

## Attack Distribution

In `Match Report -> Offensive`, attack distribution is computed from pass end locations:

- `Left`: `pass_end_location_y < 26.67`
- `Centre`: `26.67 <= pass_end_location_y <= 53.33`
- `Right`: `pass_end_location_y > 53.33`

This is shown separately for home and away teams as percentages and counts.

## Progressive Pass

A pass is progressive when it moves the ball at least `threshold` meters closer to goal.

Using StatsBomb pitch length (`120`):

- `start_dist = 120 - start_x`
- `end_dist = 120 - end_x`
- progressive if `(start_dist - end_dist) >= threshold`

UI control:

- Progressive threshold slider (`8` to `15`, default `10`)

Only **successful progressive passes** are used for top progressive passers:

- `successful_progressive = progressive AND completed`

Completion logic:

- If `pass_outcome_name` exists: pass is complete when outcome is empty/null (or `"Complete"`).
- Else if `pass_outcome_id` exists: null id is treated as complete.
- Else: completion is unavailable and passes are treated as complete, with an in-app warning.
