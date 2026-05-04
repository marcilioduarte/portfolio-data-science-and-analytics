"""Project-wide constants and feature configuration for credit risk."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "german_credit.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "Creditability"
NOT_SELECTED_LABEL = "Not selected"


@dataclass(frozen=True)
class FeatureOption:
    """Single UI option and its one-hot encoded destination column."""

    label: str
    column: str


@dataclass(frozen=True)
class FeatureGroup:
    """Feature group used in the app and preprocessing logic."""

    name: str
    source_column: str
    options: tuple[FeatureOption, ...]

    @property
    def labels(self) -> list[str]:
        return [option.label for option in self.options]

    def column_from_label(self, label: str | None) -> str | None:
        if label is None:
            return None
        for option in self.options:
            if option.label == label:
                return option.column
        return None


# This list defines both the app controls and the final model input schema.
FEATURE_GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup(
        name="Account Balance",
        source_column="Account Balance",
        options=(
            FeatureOption("No account", "Account Balance_1"),
            FeatureOption("No balance", "Account Balance_2"),
            FeatureOption("Some balance", "Account Balance_3"),
        ),
    ),
    FeatureGroup(
        name="Payment Status of Previous Credit",
        source_column="Payment Status of Previous Credit",
        options=(
            FeatureOption("Some problems", "Payment Status of Previous Credit_1"),
            FeatureOption("No problems in this bank", "Payment Status of Previous Credit_3"),
        ),
    ),
    FeatureGroup(
        name="Purpose",
        source_column="Purpose",
        options=(
            FeatureOption("New car", "Purpose_1"),
            FeatureOption("Other", "Purpose_4"),
        ),
    ),
    FeatureGroup(
        name="Value Savings/Stocks",
        source_column="Value Savings/Stocks",
        options=(
            FeatureOption("No savings", "Value Savings/Stocks_1"),
            FeatureOption("DM between [100, 1000]", "Value Savings/Stocks_3"),
            FeatureOption("DM >= 1000", "Value Savings/Stocks_5"),
        ),
    ),
    FeatureGroup(
        name="Length of Current Employment",
        source_column="Length of current employment",
        options=(
            FeatureOption("Below 1 year (or unemployed)", "Length of current employment_1"),
            FeatureOption("Between 4 and 7 years", "Length of current employment_4"),
        ),
    ),
    FeatureGroup(
        name="Instalment Per Cent",
        source_column="Instalment per cent",
        options=(FeatureOption("Smaller than 20%", "Instalment per cent_4"),),
    ),
    FeatureGroup(
        name="Guarantors",
        source_column="Guarantors",
        options=(FeatureOption("No guarantors", "Guarantors_1"),),
    ),
    FeatureGroup(
        name="Duration in Current Address",
        source_column="Duration in Current address",
        options=(
            FeatureOption("Less than a year", "Duration in Current address_1"),
            FeatureOption("Between 1 and 4 years", "Duration in Current address_2"),
        ),
    ),
    FeatureGroup(
        name="Most Valuable Available Asset",
        source_column="Most valuable available asset",
        options=(
            FeatureOption("Not available / no assets", "Most valuable available asset_1"),
            FeatureOption("Ownership of house or land", "Most valuable available asset_4"),
        ),
    ),
    FeatureGroup(
        name="Concurrent Credits",
        source_column="Concurrent Credits",
        options=(FeatureOption("No further running credits", "Concurrent Credits_3"),),
    ),
    FeatureGroup(
        name="Type of Apartment",
        source_column="Type of apartment",
        options=(FeatureOption("Free apartment", "Type of apartment_1"),),
    ),
    FeatureGroup(
        name="Number of Credits at this Bank",
        source_column="No of Credits at this Bank",
        options=(FeatureOption("One credit", "No of Credits at this Bank_1"),),
    ),
    FeatureGroup(
        name="Occupation",
        source_column="Occupation",
        options=(FeatureOption("Unemployed or unskilled with no permanent", "Occupation_1"),),
    ),
)


# Keep this explicit list to guarantee deterministic input order for training/inference.
SELECTED_FEATURES: list[str] = [
    option.column for group in FEATURE_GROUPS for option in group.options
]

