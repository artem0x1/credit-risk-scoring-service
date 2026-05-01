from pathlib import Path
import os
import shutil

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_DIR = PROJECT_ROOT / "configs"

KAGGLE_CONFIG_PATH = CONFIG_DIR / "kaggle.json"
KAGGLE_EXAMPLE_CONFIG_PATH = CONFIG_DIR / "kaggle.example.json"

COMPETITION_NAME = "home-credit-default-risk"

REQUIRED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "HomeCredit_columns_description.csv",
    "sample_submission.csv",
]


def raw_data_exists() -> bool:
    return all((RAW_DATA_DIR / filename).exists() for filename in REQUIRED_FILES)


def check_kaggle_config() -> None:
    if KAGGLE_CONFIG_PATH.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(CONFIG_DIR)
        return

    raise FileNotFoundError(
        "\nKaggle credentials were not found.\n\n"
        f"Expected file: {KAGGLE_CONFIG_PATH}\n\n"
        "Create it from the example file:\n\n"
        f"cp {KAGGLE_EXAMPLE_CONFIG_PATH} {KAGGLE_CONFIG_PATH}\n\n"
        "Then fill in your Kaggle username and API key.\n"
        "Also make sure you accepted the competition rules on Kaggle:\n"
        "https://www.kaggle.com/competitions/home-credit-default-risk/rules\n"
    )


def download_home_credit_data() -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if raw_data_exists():
        print("Raw data already exists. Skipping download.")
        return RAW_DATA_DIR

    check_kaggle_config()

    print("Downloading Home Credit Default Risk dataset...")
    kaggle_path = Path(
        kagglehub.competition_download(COMPETITION_NAME)
    )

    print(f"Kaggle cache path: {kaggle_path}")
    print(f"Copying files to: {RAW_DATA_DIR}")

    for file_path in kaggle_path.iterdir():
        target_path = RAW_DATA_DIR / file_path.name

        if target_path.exists():
            print(f"Already exists, skipping: {target_path.name}")
            continue

        if file_path.is_file():
            shutil.copy2(file_path, target_path)
            print(f"Copied: {file_path.name}")

    print("Done.")
    return RAW_DATA_DIR


if __name__ == "__main__":
    download_home_credit_data()