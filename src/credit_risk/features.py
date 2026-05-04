"""Feature preparation utilities for training and app inference."""

from __future__ import annotations

import pandas as pd

from credit_risk.config import FEATURE_GROUPS, SELECTED_FEATURES, TARGET_COLUMN


def _validate_raw_columns(df: pd.DataFrame) -> None:
    """Fail fast when required raw columns are missing."""
    required_columns = {TARGET_COLUMN, *[group.source_column for group in FEATURE_GROUPS]}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in raw dataset: {missing}")


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the model matrix from the raw CSV.

    Notes:
    - Uses one-hot encoding only on the columns represented in FEATURE_GROUPS.
    - Reindexes to SELECTED_FEATURES so train/inference always match column order.
    """
    _validate_raw_columns(df)

    categorical_columns = [group.source_column for group in FEATURE_GROUPS]
    encoded = pd.get_dummies(df[categorical_columns], columns=categorical_columns, dtype="int64")

    # Guarantee all model columns exist even if a category is absent in the current dataset.
    feature_frame = encoded.reindex(columns=SELECTED_FEATURES, fill_value=0).astype("int64")
    target = df[TARGET_COLUMN].astype("int64")
    return feature_frame, target


def build_inference_frame(selection_by_group: dict[str, str | None]) -> pd.DataFrame:
    """
    Convert app selections to a one-row DataFrame matching model schema.

    The dict format is:
      {"Account Balance": "No account", "Purpose": None, ...}
    """
    values = {column: 0 for column in SELECTED_FEATURES}

    for group in FEATURE_GROUPS:
        selected_label = selection_by_group.get(group.name)
        selected_column = group.column_from_label(selected_label)
        if selected_column is not None:
            values[selected_column] = 1

    return pd.DataFrame([values], columns=SELECTED_FEATURES, dtype="int64")

