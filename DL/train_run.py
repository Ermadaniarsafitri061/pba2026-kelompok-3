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

    # angka di atas bar
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
    model_lstm = BiLSTMClassifier(vocab_size=len(vocab))

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
    model_att = BiLSTMAttentionClassifier(vocab_size=len(vocab))

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

    print("\n✅ DONE!")


# =========================
if __name__ == "__main__":
    main()