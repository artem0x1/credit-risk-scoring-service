from pathlib import Path
from typing import Any

import yaml

from src.paths import CONFIG_PATH


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def get_config_value(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
    keys = key_path.split(".")
    value = config

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default

        value = value[key]

    return value