"""App-side helpers to load artifacts and build visual outputs."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from credit_risk.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_PATH,
    MODEL_DIR,
    REPORTS_DIR,
    SELECTED_FEATURES,
)
from credit_risk.features import build_training_frame
from credit_risk.modeling import evaluate_model, save_metrics, save_model, train_model


@dataclass
class AppArtifacts:
    """Objects loaded once at startup to keep app latency low."""

    model: Any
    metrics: dict[str, float]
    feature_importance_plot: go.Figure
    confusion_matrix_plot: go.Figure


def _load_model() -> Any:
    """Load the most recent model artifact with backward-compatible fallback."""
    joblib_path = MODEL_DIR / "model.joblib"
    legacy_pickle_path = MODEL_DIR / "model.pickle"

    if joblib_path.exists():
        try:
            return joblib.load(joblib_path)
        except Exception:
            pass

    if legacy_pickle_path.exists():
        try:
            with legacy_pickle_path.open("rb") as file:
                return pickle.load(file)
        except Exception:
            pass

    return _retrain_and_persist_artifacts()


def _retrain_and_persist_artifacts() -> Any:
    """Rebuild model artifacts when serialized files are missing/incompatible."""
    raw_df = pd.read_csv(DATA_RAW_PATH)
    features, target = build_training_frame(raw_df)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.3,
        random_state=42,
        stratify=target,
    )

    model = train_model(x_train=x_train, y_train=y_train, random_state=42)
    metrics, y_hat = evaluate_model(model=model, x_test=x_test, y_test=y_test)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    x_train.to_parquet(DATA_PROCESSED_DIR / "x_train.parquet", index=False)
    x_test.to_parquet(DATA_PROCESSED_DIR / "x_test.parquet", index=False)
    y_train.to_frame(name="target").to_parquet(DATA_PROCESSED_DIR / "y_train.parquet", index=False)
    y_test.to_frame(name="target").to_parquet(DATA_PROCESSED_DIR / "y_test.parquet", index=False)
    y_hat.to_frame(name="prediction").to_parquet(DATA_PROCESSED_DIR / "yhat.parquet", index=False)

    save_model(model=model, model_path=MODEL_DIR / "model.joblib")
    with (MODEL_DIR / "model.pickle").open("wb") as file:
        pickle.dump(model, file)
    save_metrics(metrics=metrics, path=REPORTS_DIR / "metrics.json")
    return model


def _load_metrics() -> dict[str, float]:
    """Load cached metrics, or return an empty dict when not available."""
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _load_test_outputs() -> tuple[pd.Series | None, pd.Series | None]:
    """Load y_test and yhat predictions used to generate confusion matrix."""
    y_test_path = Path("data") / "processed" / "y_test.parquet"
    y_hat_path = Path("data") / "processed" / "yhat.parquet"

    if not y_test_path.exists() or not y_hat_path.exists():
        return None, None

    y_test = pd.read_parquet(y_test_path).squeeze()
    y_hat = pd.read_parquet(y_hat_path).squeeze()
    return y_test, y_hat


def _build_feature_importance_plot(model: Any) -> go.Figure:
    """Build a robust plot even when the estimator has no feature_importances_."""
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=SELECTED_FEATURES)
        data = (
            importances.sort_values(ascending=False)
            .rename_axis("feature")
            .reset_index(name="importance")
        )
        return px.bar(
            data,
            x="feature",
            y="importance",
            title="Feature Importance",
            labels={"feature": "Feature", "importance": "Importance"},
        )

    return go.Figure(
        layout={
            "title": "Feature importance is not available for this model type.",
            "xaxis_title": "Feature",
            "yaxis_title": "Importance",
        }
    )


def _build_confusion_matrix_plot(y_test: pd.Series | None, y_hat: pd.Series | None) -> go.Figure:
    """Build confusion matrix from cached test predictions."""
    if y_test is None or y_hat is None:
        return go.Figure(
            layout={
                "title": "Confusion matrix not available yet. Run training script first.",
                "xaxis_title": "Predicted",
                "yaxis_title": "Actual",
            }
        )

    matrix = confusion_matrix(y_test, y_hat)
    return px.imshow(
        matrix,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        color_continuous_scale="Blues",
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="Confusion Matrix",
    )


def format_metrics_markdown(metrics: dict[str, float]) -> str:
    """Render metrics consistently in the UI."""
    if not metrics:
        return "Metrics not available. Run `python scripts/train_model.py` first."

    lines = ["### Model Metrics"]
    for key, value in metrics.items():
        metric_name = key.replace("_", " ").title()
        lines.append(f"- **{metric_name}:** {value:.4f}")
    return "\n".join(lines)


def load_artifacts() -> AppArtifacts:
    """Entry point used by the app to pre-load model and visual assets once."""
    model = _load_model()
    metrics = _load_metrics()
    y_test, y_hat = _load_test_outputs()

    return AppArtifacts(
        model=model,
        metrics=metrics,
        feature_importance_plot=_build_feature_importance_plot(model),
        confusion_matrix_plot=_build_confusion_matrix_plot(y_test, y_hat),
    )

