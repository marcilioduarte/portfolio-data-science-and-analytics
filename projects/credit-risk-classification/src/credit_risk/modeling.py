"""Model training and evaluation helpers for credit risk classification."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def train_model(x_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.

    A tree model keeps parity with the previous project behavior while
    remaining simple to interpret in the app via feature importances.
    """
    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=12, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: DecisionTreeClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float], pd.Series]:
    """Return classification metrics and test-set predictions."""
    y_pred = pd.Series(model.predict(x_test), index=y_test.index, name="prediction")

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    # ROC-AUC is only available when the model exposes class probabilities.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_score)), 4)

    return metrics, y_pred


def save_model(model: DecisionTreeClassifier, model_path: Path) -> None:
    """Persist model as a joblib artifact."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    """Persist metrics in JSON for app visualization and reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

