"""Utilities for app-layer inference and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from house_prices.config import (
    DATA_CLEAN_PATH,
    DATA_PROCESSED_DIR,
    DATA_RAW_PATH,
    MODEL_DIR,
    REPORTS_DIR,
)
from house_prices.features import build_inference_frame, build_training_frame
from house_prices.modeling import evaluate_model, save_metrics, save_model, train_model


def load_model() -> Any:
    """Load trained house prices model artifact."""
    model_path = MODEL_DIR / "model.joblib"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    return _retrain_and_persist_artifacts()


def load_feature_order() -> list[str]:
    """Load training feature order used by the fitted pipeline."""
    path = MODEL_DIR / "feature_order.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    source = DATA_CLEAN_PATH if DATA_CLEAN_PATH.exists() else DATA_RAW_PATH
    df = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    features, _ = build_training_frame(df)
    return list(features.columns)


def load_metrics() -> dict[str, float]:
    """Load metrics from reports directory when available."""
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _retrain_and_persist_artifacts() -> Any:
    """Rebuild model/metrics when serialized artifacts are missing or incompatible."""
    source = DATA_CLEAN_PATH if DATA_CLEAN_PATH.exists() else DATA_RAW_PATH
    df = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    features, target = build_training_frame(df)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = train_model(x_train=x_train, y_train=y_train, random_state=42)
    metrics, y_hat = evaluate_model(model=model, x_test=x_test, y_test=y_test)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    x_train.to_parquet(DATA_PROCESSED_DIR / "x_train.parquet", index=False)
    x_test.to_parquet(DATA_PROCESSED_DIR / "x_test.parquet", index=False)
    y_train.to_frame(name="target").to_parquet(DATA_PROCESSED_DIR / "y_train.parquet", index=False)
    y_test.to_frame(name="target").to_parquet(DATA_PROCESSED_DIR / "y_test.parquet", index=False)
    y_hat.to_frame(name="prediction").to_parquet(DATA_PROCESSED_DIR / "yhat.parquet", index=False)

    save_model(model=model, path=MODEL_DIR / "model.joblib")
    save_metrics(metrics=metrics, path=REPORTS_DIR / "metrics.json")
    (MODEL_DIR / "feature_order.json").write_text(json.dumps(list(x_train.columns), indent=2), encoding="utf-8")
    return model


def predict_price(model: Any, payload: dict[str, object], feature_order: list[str]) -> float:
    """Predict house price from typed payload."""
    frame: pd.DataFrame = build_inference_frame(payload=payload, feature_order=feature_order)
    prediction = model.predict(frame)[0]
    return float(prediction)


def format_metrics_markdown(metrics: dict[str, float]) -> str:
    """Format metrics for app display."""
    if not metrics:
        return "Metrics not available. Train the model first."
    lines = ["### Model Metrics"]
    for key, value in metrics.items():
        lines.append(f"- **{key.upper()}**: {value:.4f}")
    return "\n".join(lines)
