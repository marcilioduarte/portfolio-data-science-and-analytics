# California House Prices Regression

Production-style regression project adapted from the original notebook workflow.

## Objective

Predict California house prices and compare model quality using a reproducible train/evaluate pipeline.

## Project Layout

```text
data/
  raw/calif_housing_prices_.csv
  clean/calif_housing_prices_clean.parquet
src/house_prices/
  config.py
  features.py
  modeling.py
scripts/
  train_model.py
  evaluate_model.py
notebooks/
  house_prices_modeling.ipynb
```

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train and persist artifacts:

```bash
python scripts/train_model.py
```

3. Print metrics:

```bash
python scripts/evaluate_model.py
```

## Output Artifacts

- `model/model.joblib`
- `reports/metrics.json`
- `data/processed/x_train.parquet`
- `data/processed/x_test.parquet`
- `data/processed/y_train.parquet`
- `data/processed/y_test.parquet`
- `data/processed/yhat.parquet`

## Notes

- The notebook is kept as exploratory documentation.
- The reproducible pipeline lives in `src/` + `scripts/`.
