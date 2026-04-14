import gradio as gr
import os
import pandas as pd
from pycaret.classification import load_model, predict_model

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model")

model = load_model(MODEL_PATH)


def predict(text):
    df = pd.DataFrame({"clean_review": [text]})
    pred = predict_model(model, data=df)

    result = pred["prediction_label"][0]

    print("DEBUG:", result)

    if result == "positive":
        return "Positive 😊"
    else:
        return "Negative 😡"


iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="IMDB Sentiment Analysis"
)

iface.launch()