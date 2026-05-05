# Credit Risk Scoring Service

Проект представляет собой end-to-end систему кредитного скоринга на данных Home Credit Default Risk.

В проекте реализован полный ML pipeline:

* разведочный анализ данных;
* feature engineering;
* классическая scorecard-модель на WOE/IV;
* tree-based модели и LightGBM;
* сравнение моделей;
* inference pipeline;
* FastAPI сервис для получения кредитного решения.

---

## Цель проекта

Цель проекта — построить модель, которая оценивает риск дефолта клиента по кредитной заявке.

Модель возвращает:

* вероятность дефолта;
* категорию риска;
* итоговое решение: `APPROVE` или `REJECT`.

Пример ответа API:

```json
{
  "client_id": 100001,
  "model": "LightGBM",
  "probability_of_default": 0.2277,
  "threshold": 0.65,
  "risk_grade": "MEDIUM",
  "decision": "APPROVE"
}
```

---

## Структура проекта

```text
credit-risk-scoring-service/
│
├── data/
│   ├── raw/                    # исходные данные
│   ├── interim/                # промежуточные датасеты
│   └── processed/              # обработанные признаки
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_woe_iv_scorecard.ipynb
│   ├── 04_train_lightgbm.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_inference_demo.ipynb
│
├── src/
│   ├── data/                   # загрузка и подготовка данных
│   ├── features/               # feature engineering
│   ├── scoring/                # WOE, IV, binning, scorecard
│   ├── models/                 # обучение и оценка моделей
│   └── inference/              # inference pipeline
│
├── models/                     # сохранённые модели и артефакты
│
├── reports/
│   └── figures/                # графики и визуализации
│
├── api/                        # FastAPI сервис
│   ├── main.py
│   ├── schemas.py
│   ├── service.py
│   ├── settings.py
│   └── tests/
│
├── backend-java/               # Java backend-клиент для ML API
│
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Данные

Используются данные соревнования Home Credit Default Risk.

Основная целевая переменная:

```text
TARGET = 1 → клиент допустил дефолт
TARGET = 0 → клиент не допустил дефолт
```

В проекте используются следующие типы данных:

* анкетные данные клиента;
* кредитная история из bureau;
* предыдущие заявки;
* POS cash balance;
* installment payments;
* credit card balance;
* агрегированные признаки.

После feature engineering формируются файлы:

```text
data/processed/train_features.parquet
data/processed/test_features.parquet
data/processed/feature_columns.csv
```

---

## Ноутбуки

### `01_eda.ipynb`

Разведочный анализ данных:

* анализ размера датасетов;
* проверка пропусков;
* анализ целевой переменной;
* базовые распределения признаков;
* первичные выводы по данным.

---

### `02_feature_engineering.ipynb`

Построение признаков:

* обработка application data;
* агрегаты по bureau;
* агрегаты по previous applications;
* агрегаты по платежам и кредитным картам;
* объединение признаков в единый датасет;
* сохранение `train_features.parquet` и `test_features.parquet`.

---

### `03_woe_iv_scorecard.ipynb`

Классический credit scoring pipeline:

* расчёт WOE и IV;
* отбор признаков по IV;
* WOE-преобразование признаков;
* обучение логистической регрессии;
* построение scorecard;
* перевод вероятности дефолта в credit score;
* анализ score bands;
* сохранение scorecard-модели и артефактов.

Scorecard-модель используется как интерпретируемый baseline.

---

### `04_train_lightgbm.ipynb`

Обучение tree-based моделей:

* Decision Tree baseline;
* HistGradientBoosting baseline;
* LightGBM;
* сравнение моделей по ROC-AUC и PR-AUC;
* построение ROC и PR кривых;
* feature importance;
* сохранение LightGBM модели и test predictions.

Итоговая LightGBM модель показала лучшее качество среди протестированных моделей.

---

### `05_model_evaluation.ipynb`

Сравнение моделей:

* WOE Logistic Regression / Scorecard;
* LightGBM;
* ROC-AUC;
* PR-AUC;
* KS metric;
* calibration curve;
* threshold analysis;
* выбор финальной модели.

По итогам evaluation основной моделью выбран `LightGBM`.

---

### `06_inference_demo.ipynb`

Демонстрация inference pipeline:

* загрузка финальной модели;
* подготовка признаков клиента;
* prediction для одного клиента;
* batch prediction;
* risk policy;
* сохранение inference examples.

---

## Используемые модели

В проекте были протестированы:

| Модель                  | Назначение                        |
| ----------------------- | --------------------------------- |
| WOE Logistic Regression | интерпретируемая scorecard-модель |
| Decision Tree           | простой baseline                  |
| HistGradientBoosting    | tree-based baseline               |
| LightGBM                | основная ML-модель                |

Итоговое сравнение:

| Model                   | ROC-AUC | PR-AUC |
| ----------------------- | ------: | -----: |
| LightGBM                |  ~0.788 | ~0.287 |
| HistGradientBoosting    |  ~0.785 | ~0.282 |
| WOE Logistic Regression |  ~0.768 | ~0.255 |
| Decision Tree           |  ~0.727 | ~0.195 |

LightGBM выбран как финальная модель, потому что показал лучшее качество по ROC-AUC, PR-AUC, KS и Brier score.

---

## Scorecard

В проекте реализована классическая скоринговая карта:

```text
raw features → binning → WOE → Logistic Regression → scorecard points
```

Scorecard позволяет получить интерпретируемый credit score:

```text
высокий score → ниже риск дефолта
низкий score → выше риск дефолта
```

Scorecard полезна как объяснимая модель, но по качеству уступает LightGBM.

---

## LightGBM

LightGBM используется как основная ML-модель.

Преимущества:

* хорошо работает с табличными данными;
* умеет обрабатывать пропуски;
* поддерживает категориальные признаки;
* ловит нелинейности и взаимодействия признаков;
* показывает лучшее качество по сравнению с scorecard baseline.

---

## Inference pipeline

Inference pipeline реализован в модуле:

```text
src/inference/
├── preprocessor.py
├── risk_policy.py
├── predict.py
└── __init__.py
```

Основной класс:

```python
CreditRiskPredictor
```

Он умеет:

* загружать LightGBM модель;
* загружать список признаков;
* подготавливать входные данные;
* считать вероятность дефолта;
* назначать risk grade;
* возвращать кредитное решение.

Пример использования:

```python
from pathlib import Path
import pandas as pd

from src.inference import CreditRiskPredictor

root = Path.cwd()

predictor = CreditRiskPredictor(
    model_path=root / "models" / "lightgbm_model.pkl",
    feature_list_path=root / "models" / "lightgbm_feature_list.json",
    final_model_config_path=root / "models" / "final_model_config.json",
)

test = pd.read_parquet(root / "data" / "processed" / "test_features.parquet")

predictions = predictor.predict_dataframe(
    test.head(5),
    id_col="SK_ID_CURR",
)

print(predictions)
```

---

## Risk policy

Risk policy переводит вероятность дефолта в категорию риска:

|    PD range | Risk grade |
| ----------: | ---------- |
|    `< 0.20` | LOW        |
| `0.20–0.40` | MEDIUM     |
| `0.40–0.65` | HIGH       |
|   `>= 0.65` | VERY_HIGH  |

Decision rule:

```text
PD >= threshold → REJECT
PD < threshold  → APPROVE
```

Текущий threshold выбран на validation по максимальному F1-score:

```text
threshold = 0.65
```

---

## FastAPI сервис

API реализован в папке:

```text
api/
├── main.py
├── schemas.py
├── service.py
├── settings.py
└── tests/
```

Доступные endpoints:

```text
GET  /health
POST /predict
```

---

### Запуск API

Установить зависимости:

```bash
pip install fastapi uvicorn pydantic-settings
```

Запустить сервис:

```bash
uvicorn api.main:app --reload
```

Открыть Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Пример ответа:

```json
{
  "status": "ok",
  "app_name": "Credit Risk Scoring API",
  "app_version": "0.1.0"
}
```

---

### Predict endpoint

Пример request:

```json
{
  "client_id": 100001,
  "features": {
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "F",
    "AMT_INCOME_TOTAL": 135000.0,
    "AMT_CREDIT": 568800.0
  }
}
```

Важно: текущая модель ожидает полный набор подготовленных признаков, использованных при обучении.

То есть API сейчас принимает не сырые анкетные данные, а feature-engineered input.

Пример response:

```json
{
  "client_id": 100001,
  "model": "LightGBM",
  "probability_of_default": 0.22773147061797416,
  "threshold": 0.65,
  "risk_grade": "MEDIUM",
  "decision": "APPROVE"
}
```

---

## Тесты API

Запуск тестов:

```bash
pytest api/tests -v
```

Покрыты:

* `/health`;
* успешный `/predict`;
* ошибка при неполном наборе признаков.

Ожидаемый результат:

```text
3 passed
```

---

## Установка

Создать окружение:

```bash
conda create -n credit-risk python=3.12
conda activate credit-risk
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Минимальные API-зависимости:

```bash
pip install fastapi uvicorn pydantic-settings pytest httpx
```

---

## Основные артефакты

После выполнения ноутбуков создаются:

```text
models/lightgbm_model.pkl
models/lightgbm_feature_list.json
models/lightgbm_params.json
models/lightgbm_metrics.json
models/final_model_config.json
models/model_evaluation_summary.csv
models/lightgbm_feature_importance.csv
models/lightgbm_inference_predictions.csv
```

Для scorecard:

```text
models/scorecard_logreg_model.pkl
models/scorecard_woe_transformer.pkl
models/scorecard_metrics.json
models/scorecard_table.csv
models/scorecard_test_predictions.csv
```

Графики:

```text
reports/figures/lightgbm_roc_curve.png
reports/figures/lightgbm_pr_curve.png
reports/figures/lightgbm_feature_importance.png
reports/figures/model_comparison_roc_curve.png
reports/figures/model_comparison_pr_curve.png
reports/figures/model_comparison_ks_curve.png
reports/figures/model_comparison_calibration_curve.png
```

---

## Ограничения текущей версии

Текущая версия является baseline production-like pipeline.

Что пока не сделано:

* полноценный hyperparameter tuning;
* cross-validation;
* time-based validation;
* probability calibration;
* SHAP explainability;
* PSI / stability monitoring;
* business cost-based threshold optimization;
* preprocessing pipeline от сырых данных до feature-engineered input внутри API.

---

## Возможные улучшения

Следующие шаги развития проекта:

1. Добавить Optuna tuning для LightGBM.
2. Добавить cross-validation.
3. Реализовать probability calibration.
4. Добавить SHAP explanations.
5. Добавить PSI и monitoring.
6. Реализовать business threshold optimization.
7. Подключить raw-data preprocessing к API.
8. Подключить Java backend как клиент к FastAPI сервису.

---

## Итог

В проекте построена полноценная credit risk scoring система:

```text
feature engineering
→ scorecard baseline
→ LightGBM model
→ evaluation
→ inference pipeline
→ FastAPI service
```

Финальная модель `LightGBM` используется для ранжирования клиентов по риску дефолта и выдачи кредитного решения через API.
