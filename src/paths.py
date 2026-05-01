from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
LOGGING_CONFIG_PATH = CONFIG_DIR / "logging.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"

API_DIR = PROJECT_ROOT / "api"


def create_project_dirs() -> None:
    dirs = [
        CONFIG_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)