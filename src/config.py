from pathlib import Path
import yaml


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file into a dictionary."""
    config_path = Path(config_path)
    with config_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)
