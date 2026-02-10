import pandas as pd


def compute_basic_tactical_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Return simple, transparent counts as a starting point for tactical reporting."""
    if events.empty or 'type' not in events.columns:
        return pd.DataFrame(columns=['metric', 'value'])

    type_series = events['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)

    summary = [
        ('total_events', int(len(events))),
        ('passes', int((type_series == 'Pass').sum())),
        ('shots', int((type_series == 'Shot').sum())),
        ('pressures', int((type_series == 'Pressure').sum())),
        ('duels', int((type_series == 'Duel').sum())),
    ]

    return pd.DataFrame(summary, columns=['metric', 'value'])
