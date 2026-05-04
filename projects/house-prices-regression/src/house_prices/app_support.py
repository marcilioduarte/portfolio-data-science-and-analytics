"""Utilities for future app-layer inference and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from house_prices.config import MODEL_DIR, REPORTS_DIR
from house_prices.features import build_inference_frame


def load_model() -> Any:
    """Load trained house prices model artifact."""
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run `python scripts/train_model.py` first.")
    return joblib.load(model_path)


def load_metrics() -> dict[str, float]:
    """Load metrics from reports directory when available."""
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
