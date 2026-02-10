from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_bars(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a simple bar chart for quick QA and reporting scaffolding."""
    if summary_df.empty:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary_df['metric'], summary_df['value'])
    ax.set_title('Match Event Summary (Scaffold)')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
