from pathlib import Path
import json
import pandas as pd


def load_json(path: str | Path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_events(events_file: str | Path) -> pd.DataFrame:
    return pd.DataFrame(load_json(events_file))


def load_lineups(lineups_file: str | Path) -> pd.DataFrame:
    return pd.DataFrame(load_json(lineups_file))
