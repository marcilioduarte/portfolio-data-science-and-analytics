"""Gradio app for credit worthiness inference."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.app_support import format_metrics_markdown, load_artifacts  # noqa: E402
from credit_risk.config import FEATURE_GROUPS, NOT_SELECTED_LABEL  # noqa: E402
from credit_risk.features import build_inference_frame  # noqa: E402

ARTIFACTS = load_artifacts()


def predict_credit_worthiness(name: str, *selections: str) -> tuple[str, str, object, object, object]:
    """Predict loan eligibility from UI selections."""
    selection_by_group = {}
    for group, selected_label in zip(FEATURE_GROUPS, selections):
        if selected_label == NOT_SELECTED_LABEL:
            selection_by_group[group.name] = None
        else:
            selection_by_group[group.name] = selected_label

    inference_frame = build_inference_frame(selection_by_group)
    prediction = int(ARTIFACTS.model.predict(inference_frame)[0])

    if hasattr(ARTIFACTS.model, "predict_proba"):
        probabilities = ARTIFACTS.model.predict_proba(inference_frame)[0]
        confidence = max(probabilities)
    else:
        confidence = None

    user_name = (name or "there").strip()
    if prediction == 1:
        verdict = "eligible for the loan"
    else:
        verdict = "not eligible for the loan at the moment"

    if confidence is None:
        prediction_text = f"Hi {user_name}. According to the model, your client is {verdict}."
    else:
        prediction_text = (
            f"Hi {user_name}. According to the model, your client is {verdict}. "
            f"(Confidence: {confidence:.2%})"
        )

    return (
        prediction_text,
        format_metrics_markdown(ARTIFACTS.metrics),
        ARTIFACTS.feature_importance_plot,
        ARTIFACTS.confusion_matrix_plot,
        ARTIFACTS.roc_curve_plot,
    )


with gr.Blocks(title="Credit Worthiness Risk Classification") as demo:
    gr.Markdown("# Credit Worthiness Risk Classification")
    gr.Markdown(
        "Select the option that best describes the client in each section, then run prediction."
    )

    name_input = gr.Textbox(label="Analyst Name", placeholder="Your name")
    selection_components = []

    with gr.Accordion("Client Profile Inputs", open=True):
        for group in FEATURE_GROUPS:
            with gr.Row():
                component = gr.Radio(
                    choices=[NOT_SELECTED_LABEL, *group.labels],
                    value=NOT_SELECTED_LABEL,
                    label=group.name,
                )
                selection_components.append(component)

    predict_button = gr.Button("Predict")
    prediction_output = gr.Textbox(label="Prediction", interactive=False)
    metrics_output = gr.Markdown(label="Metrics")
    feature_plot_output = gr.Plot(label="Feature Importance")
    matrix_plot_output = gr.Plot(label="Confusion Matrix")
    roc_plot_output = gr.Plot(label="ROC Curve")

    predict_button.click(
        fn=predict_credit_worthiness,
        inputs=[name_input, *selection_components],
        outputs=[prediction_output, metrics_output, feature_plot_output, matrix_plot_output, roc_plot_output],
    )

demo.launch()
