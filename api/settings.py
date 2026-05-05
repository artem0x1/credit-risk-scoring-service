from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """
    API settings.

    Paths are resolved relative to project root.
    """

    model_config = SettingsConfigDict(
        env_prefix="CREDIT_SCORING_",
    )

    app_name: str = "Credit Risk Scoring API"
    app_version: str = "0.1.0"

    model_path: Path = PROJECT_ROOT / "models" / "lightgbm_model.pkl"
    feature_list_path: Path = PROJECT_ROOT / "models" / "lightgbm_feature_list.json"
    final_model_config_path: Path = PROJECT_ROOT / "models" / "final_model_config.json"


settings = Settings()