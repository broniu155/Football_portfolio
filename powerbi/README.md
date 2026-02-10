# Power BI Handoff

This project exports a star schema to `data_processed/` for direct Power BI import.

## Exported Tables

- `data_processed/fact_events.csv`
- `data_processed/dim_match.csv`
- `data_processed/dim_team.csv`
- `data_processed/dim_player.csv`

All columns are flat scalar values (no nested arrays/dicts).

## Grain And Keys

- `fact_events`: one row per event (`event_id`) with stable foreign keys:
  - `match_id`
  - `team_id`
  - `player_id`
- `dim_match`: one row per `match_id`
- `dim_team`: one row per `team_id`
- `dim_player`: one row per `player_id`

## Relationship Diagram (Description)

Use a classic star with `fact_events` in the center:

1. `dim_match[match_id]` (1) -> `fact_events[match_id]` (*)
2. `dim_team[team_id]` (1) -> `fact_events[team_id]` (*)
3. `dim_player[player_id]` (1) -> `fact_events[player_id]` (*)

Set filter direction to single direction from dimensions to fact for predictable DAX behavior.

## Recommended Base Measures

Create these measures first:

```DAX
Events = COUNTROWS(fact_events)
```

```DAX
Passes =
CALCULATE(
    [Events],
    fact_events[type_name] = "Pass"
)
```

```DAX
Shots =
CALCULATE(
    [Events],
    fact_events[type_name] = "Shot"
)
```

```DAX
Total xG =
SUM(fact_events[shot_statsbomb_xg])
```

```DAX
Pass Completion % =
DIVIDE(
    CALCULATE([Passes], ISBLANK(fact_events[pass_outcome_name])),
    [Passes]
)
```

```DAX
Defensive Actions Opp Half =
CALCULATE(
    [Events],
    fact_events[type_name] IN {
        "Pressure",
        "Duel",
        "Interception",
        "Block",
        "Ball Recovery",
        "Clearance",
        "Foul Committed"
    },
    fact_events[location_x] >= 60
)
```

## Refresh Flow

1. Generate processed events (`src/etl.py`).
2. Export star schema:
   - `py -3 src/export_star_schema.py`
3. Refresh Power BI model from `data_processed/*.csv`.
