"""Train and persist the house prices regression model and artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from house_prices.config import (  # noqa: E402
    DATA_CLEAN_PATH,
    DATA_PROCESSED_DIR,
    DATA_RAW_PATH,
    MODEL_DIR,
    REPORTS_DIR,
)
from house_prices.features import build_training_frame  # noqa: E402
from house_prices.modeling import evaluate_model, save_metrics, save_model, train_model  # noqa: E402


def _load_source_dataframe() -> pd.DataFrame:
    if DATA_CLEAN_PATH.exists():
        return pd.read_parquet(DATA_CLEAN_PATH)
    return pd.read_csv(DATA_RAW_PATH)


def main() -> None:
    df = _load_source_dataframe()
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

    print("Training finished successfully.")
    print(f"Saved model to: {MODEL_DIR / 'model.joblib'}")
    print(f"Saved metrics to: {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
