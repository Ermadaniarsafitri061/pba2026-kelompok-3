import gradio as gr
import torch
import json
from models import BiLSTMClassifier

# load vocab
with open("vocab.json") as f:
    word2idx = json.load(f)

# load model
model = BiLSTMClassifier(
    vocab_size=len(word2idx),
    embed_dim=128,
    hidden_dim=128
)

model.load_state_dict(torch.load("bilstm_model.pt", map_location="cpu"))
model.eval()

def preprocess(text, max_len=80):
    tokens = text.lower().split()
    indices = [word2idx.get(t, 1) for t in tokens][:max_len]
    indices += [0] * (max_len - len(indices))
    return torch.tensor([indices]), torch.tensor([len(tokens)])

def predict(text):
    x, l = preprocess(text)
    with torch.no_grad():
        output = model(x, l)
        pred = output.argmax(dim=1).item()

    return "Positive 😊" if pred == 1 else "Negative 😡"


gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Text(label="Prediction"),
    title="Sentiment Analysis BiLSTM",
    description="Model NLP menggunakan BiLSTM"
).launch()