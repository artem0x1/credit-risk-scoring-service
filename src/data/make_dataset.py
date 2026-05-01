import pandas as pd

from src.paths import INTERIM_DATA_DIR
from src.data.load_data import load_application_train, load_application_test
from src.data.validate_data import validate_application_train, validate_application_test


def make_base_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    application_train = load_application_train()
    application_test = load_application_test()

    validate_application_train(application_train)
    validate_application_test(application_test)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = INTERIM_DATA_DIR / "application_train.parquet"
    test_path = INTERIM_DATA_DIR / "application_test.parquet"

    application_train.to_parquet(train_path, index=False)
    application_test.to_parquet(test_path, index=False)

    print(f"Saved train dataset to: {train_path}")
    print(f"Saved test dataset to: {test_path}")

    return application_train, application_test


if __name__ == "__main__":
    make_base_datasets()