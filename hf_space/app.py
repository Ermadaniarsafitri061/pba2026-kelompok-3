import warnings
warnings.filterwarnings("ignore")

import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load model (tanpa .pkl)
model = load_model("best_model")


def predict_sentiment(text):
    if not text or not text.strip():
        return {"Empty Input": 1.0}

    df_input = pd.DataFrame({
        "clean_review": [text]
    })

    result = predict_model(model, data=df_input)

    # Ambil label
    label = result.get("prediction_label", result.iloc[:, -1])[0]

    # Ambil confidence
    score = result.get("prediction_score", pd.Series([1.0]))[0]

    return {str(label): float(score)}


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Masukkan Review Film",
        placeholder="Contoh: This movie was amazing!",
        lines=3,
    ),
    outputs=gr.Label(num_top_classes=2),
    title="🎬 IMDB Sentiment Analysis",
    description="Sentiment Analysis menggunakan PyCaret",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()