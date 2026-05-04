"""Model training and evaluation utilities for house prices regression."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in x_train.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def train_model(x_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> GridSearchCV:
    """Train a RandomForest model wrapped in GridSearchCV."""
    pipeline = Pipeline(
        [
            ("preprocessor", _build_preprocessor(x_train)),
            ("regressor", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [None, 20],
            "regressor__min_samples_split": [2, 5],
        },
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def evaluate_model(model: GridSearchCV, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float], pd.Series]:
    """Compute core regression metrics and return predictions."""
    y_hat = pd.Series(model.predict(x_test), index=y_test.index)

    mse = mean_squared_error(y_test, y_hat)
    metrics = {
        "rmse": round(float(mse**0.5), 4),
        "mae": round(float(mean_absolute_error(y_test, y_hat)), 4),
        "r2": round(float(r2_score(y_test, y_hat)), 4),
    }
    return metrics, y_hat


def save_model(model: GridSearchCV, path: Path) -> None:
    """Persist trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    """Persist metrics to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
