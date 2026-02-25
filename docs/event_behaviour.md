# Event Behaviour Layer

This project now derives an additive event-behaviour layer for Match Report from `fact_events`.

## Buckets

- `OFFENSIVE`: pass/carry/dribble/shot/ball receipt actions.
- `DEFENSIVE`: pressure/interception/block/clearance/duel/recovery/foul/goalkeeper defensive actions.
- `TRANSITION`: turnovers (`Miscontrol`, `Dispossessed`, `Error`), incomplete ball receipts, and counter/counterpress context.
- `SET_PIECE`: any event tagged by set-piece play-pattern context.
- `OTHER`: uncategorized events.

Priority is strict:
1. `SET_PIECE`
2. `TRANSITION`
3. base type mapping (`OFFENSIVE`/`DEFENSIVE`)
4. `OTHER`

## Derived Fields

`derive_event_labels()` adds:

- `bucket`
- `subtype`
- `is_under_pressure`
- `is_counterpress`
- `is_set_piece`
- `is_counter`
- `is_turnover`
- `is_regain`
- `pass_height` (canonical name-friendly pass height)
- `pass_outcome` (canonical name-friendly pass outcome)

Optional helper:

- `derive_counterpress_regains(window_seconds=6.0)` flags `is_counterpress_regain` when the team losing the ball regains it quickly.

## Config

Base bucket mapping is config-driven in:

- `app/assets/event_buckets.yml`

To extend behavior, add event type names under `offensive`, `defensive`, or `transition`.
Set-piece and transition priority rules still apply on top of config mapping.
