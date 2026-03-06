from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.components.data import get_active_data_mode, load_match_events
from app.components.set_piece_data import SET_PIECE_OUTPUT_COLUMNS, extract_set_piece_events


DEFAULT_EVENT_COLUMNS = [
    "event_id",
    "match_id",
    "event_index",
    "period",
    "minute",
    "second",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "type_name",
    "play_pattern_name",
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
    "shot_end_location_x",
    "shot_end_location_y",
    "pass_outcome_name",
    "pass_outcome",
    "shot_outcome_name",
    "shot_outcome",
    "pass_cross",
    "pass_recipient_name",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate set-piece extraction/enrichment output.")
    parser.add_argument("--match-id", type=int, required=True, help="Match ID to validate.")
    parser.add_argument("--data-mode", type=str, default="", help="Data mode override (sample/remote/local_generated).")
    parser.add_argument("--sample-rows", type=int, default=10, help="How many rows to print.")
    parser.add_argument("--follow-up-seconds", type=int, default=15)
    parser.add_argument("--next-n-actions", type=int, default=5)
    args = parser.parse_args()

    mode = str(args.data_mode).strip().lower() if str(args.data_mode).strip() else get_active_data_mode()
    events = load_match_events(match_id=int(args.match_id), data_mode=mode, columns=DEFAULT_EVENT_COLUMNS)
    out = extract_set_piece_events(
        events,
        include_follow_up=True,
        follow_up_seconds=int(args.follow_up_seconds),
        next_n_actions=int(args.next_n_actions),
    )

    print(f"Data mode: {mode}")
    print(f"Input events: {len(events):,}")
    print(f"Set-piece rows: {len(out):,}")
    print("\nSchema:")
    print(", ".join(out.columns.tolist()))
    missing = [c for c in SET_PIECE_OUTPUT_COLUMNS if c not in out.columns]
    if missing:
        print(f"\n[WARN] Missing expected columns: {missing}")
    else:
        print("\n[PASS] Output contains expected columns.")

    print(f"\nSample rows (top {int(args.sample_rows)}):")
    if out.empty:
        print("(no set-piece rows found)")
        return
    preview = out.head(int(args.sample_rows)).copy()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
