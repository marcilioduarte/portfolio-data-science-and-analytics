"""Gradio app for California house price prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from house_prices.app_support import (  # noqa: E402
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
) -> tuple[str, str]:
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
    return prediction_text, format_metrics_markdown(METRICS)


with gr.Blocks(title="California House Prices Regression") as demo:
    gr.Markdown("# California House Prices Regression")
    gr.Markdown("Provide property and location features to estimate median house value.")

    with gr.Row():
        longitude = gr.Number(label="Longitude", value=-122.23)
        latitude = gr.Number(label="Latitude", value=37.88)
        housing_median_age = gr.Number(label="Housing Median Age", value=41)

    with gr.Row():
        total_rooms = gr.Number(label="Total Rooms", value=880)
        total_bedrooms = gr.Number(label="Total Bedrooms", value=129)
        population = gr.Number(label="Population", value=322)

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
        outputs=[prediction_output, metrics_output],
    )

demo.launch()
