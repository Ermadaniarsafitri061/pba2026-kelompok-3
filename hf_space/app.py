"""
app.py — Gradio App IMDB Sentiment Analysis
============================================
Deploy untuk Hugging Face Spaces.
Model: best_model.pkl (PyCaret Classification)
"""

import warnings
warnings.filterwarnings("ignore")

import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

model = load_model("best_model")


def predict_sentiment(text: str):
    if not text or not text.strip():
        return {"Input kosong": 1.0}

    # Harus sama dengan kolom saat training
    df_input = pd.DataFrame({
        "clean_review": [text]
    })

    result = predict_model(model, data=df_input)

    # Ambil label
    if "prediction_label" in result.columns:
        label = result["prediction_label"].iloc[0]
    else:
        label = result.iloc[0, -1]

    # Ambil confidence jika ada
    if "prediction_score" in result.columns:
        score = result["prediction_score"].iloc[0]
    else:
        score = 1.0

    return {str(label): float(score)}


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Masukkan Review Film",
        placeholder="Contoh: This movie was amazing!",
        lines=3,
    ),
    outputs=gr.Label(num_top_classes=2),
    title="IMDB Sentiment Analysis",
    description="Model klasifikasi sentimen Positive / Negative menggunakan PyCaret.",
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()