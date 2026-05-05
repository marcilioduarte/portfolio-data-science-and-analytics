# Data Science and Analytics Portfolio

Professional portfolio with reproducible machine learning projects for
classification and regression workflows, including Hugging Face Space apps.

## Repository Structure

```
portfolio-data-science-and-analytics/
  projects/
    credit-risk-classification/
    house-prices-regression/
```

Each project contains:

- dataset snapshots (`data/`)
- reproducible training scripts (`scripts/`)
- modular Python code (`src/`)
- project-specific documentation (`README.md`)

## Projects

### 1) Credit Risk Classification

Location: `projects/credit-risk-classification`

- Binary classification for loan eligibility.
- End-to-end flow: preprocessing, training, evaluation, and Gradio app.
- Includes feature importance and confusion matrix visual outputs.
- Deployed as an interactive Hugging Face Space app.

### 2) House Prices Regression

Location: `projects/house-prices-regression`

- Regression case study using California housing data.
- Includes feature engineering, model comparison, and hyperparameter tuning.
- Produces metrics and trained model artifact for reproducible evaluation.
- Deployed as an interactive Hugging Face Space app.

## Quick Start

1. Create and activate a virtual environment.
2. Install each project's dependencies using its local `requirements.txt`.
3. Follow the execution commands in each project's README.

## Quality Controls

- Linting with `ruff`.
- Unit tests for critical preprocessing and feature-building logic.
- GitHub Actions workflow for CI validation on push and pull request.

## Space Deploy

- Reusable script: `scripts/deploy_hf_space.ps1`
- Purpose: deploy a clean project snapshot from `projects/` to a Hugging Face Space.
