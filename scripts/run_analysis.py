from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.data_loader import load_events
from src.features.tactical_metrics import compute_basic_tactical_summary
from src.visualization.plots import plot_metric_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run football tactical analytics scaffold pipeline.')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file.')
    return parser.parse_args()


def find_first_events_file(data_root: Path) -> Path:
    events_dir = data_root / 'events'
    files = sorted(events_dir.glob('*.json'))
    if not files:
        raise FileNotFoundError(f'No event files found in {events_dir}')
    return files[0]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_root = Path(cfg['data']['root'])
    figures_dir = Path(cfg['outputs']['figures_dir'])
    tables_dir = Path(cfg['outputs']['tables_dir'])
    tables_dir.mkdir(parents=True, exist_ok=True)

    events_file = find_first_events_file(data_root)
    events = load_events(events_file)
    summary = compute_basic_tactical_summary(events)

    summary.to_csv(tables_dir / 'match_summary_scaffold.csv', index=False)
    plot_metric_bars(summary, figures_dir / 'match_summary_scaffold.png')

    print(f'Processed events file: {events_file.name}')
    print(f'Wrote table: {tables_dir / "match_summary_scaffold.csv"}')
    print(f'Wrote figure: {figures_dir / "match_summary_scaffold.png"}')


if __name__ == '__main__':
    main()
