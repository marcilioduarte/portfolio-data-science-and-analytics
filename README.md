# Credit Worthiness Risk Classification

End-to-end classification case study for credit risk prediction, including preprocessing, model selection, and app delivery.

## Objective

Build and evaluate models to classify credit applicants by risk profile in a reproducible workflow.

## What Is Included

- data preparation and feature engineering steps;
- supervised models: Logistic Regression, Decision Tree, Random Forest;
- hyperparameter tuning with `GridSearchCV`;
- model evaluation pipeline;
- lightweight Gradio app for interactive inference (`app.py`).

## Dataset

- Local path: `data/raw/german_credit.csv`
- Source: [Kaggle - German Credit Dataset](https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsgermancsv)

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Project Metadata

- `sdk`: gradio
- `sdk_version`: 3.27.0
- `app_file`: `app.py`
- `license`: Apache-2.0
