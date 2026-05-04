"""Feature preparation for house prices regression."""

from __future__ import annotations

import pandas as pd

from house_prices.config import TARGET_COLUMN


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into model features and target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    features = df.drop(columns=[TARGET_COLUMN]).copy()
    target = df[TARGET_COLUMN].copy()
    return features, target
