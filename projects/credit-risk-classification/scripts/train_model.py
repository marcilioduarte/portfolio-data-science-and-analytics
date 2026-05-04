"""Train and persist the credit risk model and evaluation artifacts."""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.config import DATA_PROCESSED_DIR, DATA_RAW_PATH, MODEL_DIR, REPORTS_DIR
from credit_risk.features import build_training_frame
from credit_risk.modeling import evaluate_model, save_metrics, save_model, train_model


def main() -> None:
    """Main training flow used by local runs and CI validations."""
    raw_df = pd.read_csv(DATA_RAW_PATH)
    features, target = build_training_frame(raw_df)

    # Stratification preserves class balance between train and test sets.
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
    # Backward-compatible artifact used by the existing Gradio Space app.
    with (MODEL_DIR / "model.pickle").open("wb") as file:
        pickle.dump(model, file)
    save_metrics(metrics=metrics, path=REPORTS_DIR / "metrics.json")

    print("Training finished successfully.")
    print(f"Saved model to: {MODEL_DIR / 'model.joblib'}")
    print(f"Saved metrics to: {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
