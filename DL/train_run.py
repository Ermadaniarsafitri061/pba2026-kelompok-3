import torch
import config
print(config.__file__)

from config import *
from preprocess import preprocess
from dataset import Vocabulary, get_lstm_dataloaders
from models import (
    BiLSTMClassifier,
    BiLSTMAttentionClassifier,
)
from train import (
    set_seed,
    train_model,
)

import matplotlib.pyplot as plt
import numpy as np
import os


# =========================
# PLOT MODEL COMPARISON
# =========================
def plot_model_comparison(results):
    models = list(results.keys())
    acc = [results[m]["accuracy"] for m in models]
    f1_macro = [results[m]["f1_macro"] for m in models]
    f1_weighted = [results[m]["f1_weighted"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10,6))

    plt.bar(x - width, acc, width, label="Accuracy")
    plt.bar(x, f1_macro, width, label="F1 Macro")
    plt.bar(x + width, f1_weighted, width, label="F1 Weighted")

    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Komparasi Model Deep Learning")
    plt.legend()

    for i in range(len(models)):
        plt.text(x[i] - width, acc[i], f"{acc[i]:.3f}", ha='center')
        plt.text(x[i], f1_macro[i], f"{f1_macro[i]:.3f}", ha='center')
        plt.text(x[i] + width, f1_weighted[i], f"{f1_weighted[i]:.3f}", ha='center')

    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, "model_comparison.png")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"📊 Saved: {path}")


# =========================
# 🔥 SAVE UNTUK VISUALISASI (DL)
# =========================
def save_dl_predictions(model, test_loader):
    print("\n💾 Menyimpan hasil DL untuk visualisasi...")

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            # binary classification → pakai sigmoid
            probs = torch.sigmoid(outputs).squeeze()

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_proba = np.array(all_preds)
    y_test = np.array(all_labels)

    # pastikan folder results ada
    os.makedirs("../results", exist_ok=True)

    np.save("../results/y_test.npy", y_test)
    np.save("../results/y_proba.npy", y_proba)

    print("✅ DL: y_test & y_proba berhasil disimpan!")


# =========================
# MAIN PIPELINE
# =========================
def main():
    set_seed(RANDOM_SEED)

    print("Device:", DEVICE)

    # =========================
    # PREPROCESS
    # =========================
    print("\nLoading & preprocessing...")
    df = preprocess()
    print(f"✅ Data loaded: {len(df)} samples")

    # =========================
    # VOCAB
    # =========================
    print("\nBuilding vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocab(df["clean_review"].tolist(), max_size=VOCAB_SIZE)
    print(f"📚 Vocabulary size: {len(vocab)}")

    # =========================
    # DATALOADER
    # =========================
    train_loader, val_loader, test_loader = get_lstm_dataloaders(
        df,
        vocab
    )

    # =========================
    # TRAIN BiLSTM
    # =========================
    print("\n🚀 Training BiLSTM...")
    model_lstm = BiLSTMClassifier(vocab_size=len(vocab)).to(DEVICE)

    metrics_lstm = train_model(
        model_lstm,
        train_loader,
        val_loader,
        test_loader,
        save_path=BILSTM_MODEL_PATH,
        epochs=LSTM_EPOCHS,
        lr=LSTM_LR,
        patience=LSTM_PATIENCE,
        model_name="lstm"
    )

    # =========================
    # TRAIN BiLSTM ATTENTION
    # =========================
    print("\n🚀 Training BiLSTM + Attention...")
    model_att = BiLSTMAttentionClassifier(vocab_size=len(vocab)).to(DEVICE)

    metrics_att = train_model(
        model_att,
        train_loader,
        val_loader,
        test_loader,
        save_path=BILSTM_ATT_MODEL_PATH,
        epochs=LSTM_EPOCHS,
        lr=LSTM_LR,
        patience=LSTM_PATIENCE,
        model_name="lstm_attention"
    )

    # =========================
    # MODEL COMPARISON
    # =========================
    print("\n📊 Comparing models...")

    results = {
        "BiLSTM": metrics_lstm,
        "BiLSTM+Attention": metrics_att,
    }

    plot_model_comparison(results)

    # =========================
    # 🔥 SAVE PREDICTION UNTUK VISUALISASI
    # =========================
    print("\n💾 Saving predictions for visualization...")

    # pakai model terbaik (BiLSTM+Attention)
    save_dl_predictions(model_att, test_loader)

    print("\n✅ DONE!")


# =========================
if __name__ == "__main__":
    main()