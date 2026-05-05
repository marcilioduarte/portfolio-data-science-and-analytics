---
title: California House Prices Regression
emoji: "🏠"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

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
  app_support.py
  config.py
  features.py
  modeling.py
scripts/
  train_model.py
  evaluate_model.py
app.py
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

4. Run app:

```bash
python app.py
```

## Output Artifacts

- `model/model.joblib` (generated at runtime/local training)
- `model/feature_order.json` (generated at runtime/local training)
- `reports/metrics.json` (generated at runtime/local training)
- `data/processed/*.parquet` (generated at runtime/local training)

## Notes

- The notebook is kept as exploratory documentation.
- The reproducible pipeline lives in `src/` + `scripts/`.
- `src/house_prices/app_support.py` is ready for the upcoming Gradio app layer.
- Binary artifacts are intentionally not versioned to keep Hugging Face Space deploys compatible.
