"""Utilities for app-layer inference and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
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

_REFERENCE_DF: pd.DataFrame | None = None


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
    if "rmse" in metrics:
        lines.append(
            f"- **RMSE:** {metrics['rmse']:.4f}  \n"
            "  Root Mean Squared Error. Penalizes larger prediction errors more strongly. "
            "Important for risk-sensitive pricing use cases. Lower is better."
        )
    if "mae" in metrics:
        lines.append(
            f"- **MAE:** {metrics['mae']:.4f}  \n"
            "  Mean Absolute Error. Average absolute prediction error in target units (USD). "
            "Important for business interpretability. Lower is better."
        )
    if "r2" in metrics:
        lines.append(
            f"- **R²:** {metrics['r2']:.4f}  \n"
            "  Proportion of target variance explained by the model. "
            "Important for overall explanatory power. Closer to 1.0 is better."
        )
    return "\n".join(lines)


def build_location_map(latitude: float, longitude: float, predicted_value: float) -> go.Figure:
    """Create a California-focused map with the prediction point marker."""
    fig = go.Figure(
        go.Scattermapbox(
            lat=[latitude],
            lon=[longitude],
            mode="markers",
            marker={"size": 12},
            text=[f"Estimated Value: ${predicted_value:,.2f}"],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=f"Prediction Location (Estimated Value: ${predicted_value:,.2f})",
        mapbox={
            "style": "open-street-map",
            # Fixed center/zoom to keep a California-focused viewport.
            "center": {"lat": 36.7783, "lon": -119.4179},
            "zoom": 4.8,
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    return fig


def _load_reference_dataframe() -> pd.DataFrame:
    """Load dataset used to recover nearest district features from coordinates."""
    global _REFERENCE_DF
    if _REFERENCE_DF is not None:
        return _REFERENCE_DF

    source = DATA_RAW_PATH
    frame = pd.read_csv(source)
    required = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]
    _REFERENCE_DF = frame[required].dropna().reset_index(drop=True)
    return _REFERENCE_DF


def geocode_california_address(address: str) -> tuple[float, float, str]:
    """Geocode address constrained to California, USA."""
    query = f"{address}, California, USA"
    geocoder = Nominatim(user_agent="house-prices-regression-app")
    location = geocoder.geocode(query, country_codes="us", addressdetails=False, exactly_one=True)
    if location is None:
        raise ValueError("Address not found. Try a more specific California address.")
    return float(location.latitude), float(location.longitude), str(location.address)


def nearest_district_profile(latitude: float, longitude: float) -> dict[str, object]:
    """Return nearest dataset district profile for a given coordinate pair."""
    frame = _load_reference_dataframe()
    dlon = frame["longitude"] - float(longitude)
    dlat = frame["latitude"] - float(latitude)
    idx = ((dlon * dlon) + (dlat * dlat)).idxmin()
    row = frame.loc[int(idx)]
    return {
        "longitude": float(row["longitude"]),
        "latitude": float(row["latitude"]),
        "housing_median_age": float(row["housing_median_age"]),
        "total_rooms": float(row["total_rooms"]),
        "total_bedrooms": float(row["total_bedrooms"]),
        "population": float(row["population"]),
        "households": float(row["households"]),
        "median_income": float(row["median_income"]),
        "ocean_proximity": str(row["ocean_proximity"]),
    }


def autofill_profile_from_address(address: str) -> tuple[dict[str, object], str]:
    """
    Build an input profile from address by:
    1) geocoding the address in California,
    2) finding nearest district record in the reference dataset.
    """
    latitude, longitude, resolved_address = geocode_california_address(address)
    profile = nearest_district_profile(latitude=latitude, longitude=longitude)
    status = (
        f"Address resolved to: {resolved_address}. "
        "District features were populated from the nearest available dataset district."
    )
    return profile, status
