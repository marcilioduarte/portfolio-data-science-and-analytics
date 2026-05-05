"""Gradio app for California house price prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from house_prices.app_support import (  # noqa: E402
    autofill_profile_from_address,
    build_location_map,
    format_metrics_markdown,
    load_feature_order,
    load_metrics,
    load_model,
    predict_price,
)

MODEL = load_model()
FEATURE_ORDER = load_feature_order()
METRICS = load_metrics()


def run_prediction(
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str,
) -> tuple[str, str, object]:
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }

    predicted_value = predict_price(model=MODEL, payload=payload, feature_order=FEATURE_ORDER)
    prediction_text = f"Estimated median house value: ${predicted_value:,.2f}"
    map_plot = build_location_map(latitude=latitude, longitude=longitude, predicted_value=predicted_value)
    return prediction_text, format_metrics_markdown(METRICS), map_plot


def autofill_from_address(address: str) -> tuple[float, float, float, float, float, float, float, float, str, str]:
    """Autofill model inputs from a California address."""
    profile, status = autofill_profile_from_address(address)
    return (
        profile["longitude"],
        profile["latitude"],
        profile["housing_median_age"],
        profile["total_rooms"],
        profile["total_bedrooms"],
        profile["population"],
        profile["households"],
        profile["median_income"],
        profile["ocean_proximity"],
        status,
    )


with gr.Blocks(title="California House Prices Regression") as demo:
    gr.Markdown("# California House Prices Regression")
    gr.Markdown("Provide property and location features to estimate median house value.")
    with gr.Accordion("Address Autofill", open=False):
        gr.Markdown(
            "You can enter a California address to auto-populate district features "
            "from the nearest district available in the training dataset."
        )
        address_input = gr.Textbox(label="California Address", placeholder="e.g., 1600 Amphitheatre Pkwy, Mountain View")
        autofill_button = gr.Button("Autofill From Address")
        autofill_status = gr.Markdown()
    with gr.Accordion("Variable Guide", open=False):
        gr.Markdown(
            "- **Longitude / Latitude:** Geographic coordinates of the district.\n"
            "- **Housing Median Age:** Median age of houses in the area.\n"
            "- **Total Rooms:** Total number of rooms in the district.\n"
            "- **Total Bedrooms:** Total number of bedrooms in the district.\n"
            "- **District Population:** Total population in the district.\n"
            "- **Households:** Number of households in the district.\n"
            "- **Median Income:** Median household income in the district.\n"
            "- **Ocean Proximity:** Categorical location profile relative to the coast."
        )

    with gr.Row():
        longitude = gr.Number(label="Longitude", value=-122.23)
        latitude = gr.Number(label="Latitude", value=37.88)
        housing_median_age = gr.Number(label="Housing Median Age", value=41)

    with gr.Row():
        total_rooms = gr.Number(label="Total Rooms", value=880)
        total_bedrooms = gr.Number(label="Total Bedrooms", value=129)
        population = gr.Number(label="District Population", value=322)

    with gr.Row():
        households = gr.Number(label="Households", value=126)
        median_income = gr.Number(label="Median Income", value=8.3252)
        ocean_proximity = gr.Dropdown(
            choices=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
            value="NEAR BAY",
            label="Ocean Proximity",
        )

    predict_button = gr.Button("Predict Price")
    prediction_output = gr.Textbox(label="Prediction", interactive=False)
    metrics_output = gr.Markdown(label="Metrics")
    map_output = gr.Plot(label="Location Map")

    autofill_button.click(
        fn=autofill_from_address,
        inputs=[address_input],
        outputs=[
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,
            ocean_proximity,
            autofill_status,
        ],
    )

    predict_button.click(
        fn=run_prediction,
        inputs=[
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,
            ocean_proximity,
        ],
        outputs=[prediction_output, metrics_output, map_output],
    )

demo.launch()
