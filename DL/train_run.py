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
    evaluate,
    print_report,
    plot_confusion,
)

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

    train_model(
        model_lstm,
        train_loader,
        val_loader,
        model_type="lstm",
        save_path=BILSTM_MODEL_PATH,
        epochs=LSTM_EPOCHS,
        lr=LSTM_LR,
        patience=LSTM_PATIENCE,
    )

    # =========================
    # TRAIN BiLSTM ATTENTION
    # =========================
    print("\n🚀 Training BiLSTM + Attention...")
    model_att = BiLSTMAttentionClassifier(vocab_size=len(vocab))

    train_model(
        model_att,
        train_loader,
        val_loader,
        model_type="lstm",
        save_path=BILSTM_ATT_MODEL_PATH,
        epochs=LSTM_EPOCHS,
        lr=LSTM_LR,
        patience=LSTM_PATIENCE,
    )

    # =========================
    # EVALUATION
    # =========================
    print("\n📊 Evaluating...")

    criterion = torch.nn.CrossEntropyLoss()

    _, _, preds, labels = evaluate(model_lstm, test_loader, criterion)
    print("\n📌 BiLSTM:")
    print_report(labels, preds)
    plot_confusion(labels, preds)

    _, _, preds, labels = evaluate(model_att, test_loader, criterion)
    print("\n📌 BiLSTM + Attention:")
    print_report(labels, preds)
    plot_confusion(labels, preds)

    print("\n✅ DONE!")


# =========================
if __name__ == "__main__":
    main()