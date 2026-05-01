import logging
import logging.config
from pathlib import Path

import yaml

from src.paths import LOGGING_CONFIG_PATH, LOGS_DIR


def setup_logger(name: str = "credit_scoring") -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if LOGGING_CONFIG_PATH.exists():
        with open(LOGGING_CONFIG_PATH, "r", encoding="utf-8") as file:
            logging_config = yaml.safe_load(file)

        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    return logging.getLogger(name)