"""Project configuration for the house prices regression workflow."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "calif_housing_prices_.csv"
DATA_CLEAN_PATH = PROJECT_ROOT / "data" / "clean" / "calif_housing_prices_clean.parquet"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "median_house_value"
