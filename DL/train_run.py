import os
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast

from config import *
from dataset import Vocabulary, get_lstm_dataloaders, get_bert_dataloaders
from models import (
    BiLSTMClassifier,
    BiLSTMAttentionClassifier,
    DistilBERTClassifier,
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
    # LOAD CLEAN DATA (FIXED)
    # =========================
    print("\nLoading clean dataset...")

    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(
            f"❌ File tidak ditemukan: {CLEAN_DATA_PATH}\n"
        )

    df = pd.read_csv(CLEAN_DATA_PATH)

    # pastikan label_encoded ada
    if "label_encoded" not in df.columns:
        df["label_encoded"] = df[LABEL_COL].map(LABEL_MAPPING)

    print(f"✅ Data loaded: {len(df)} samples")

    # =========================
    # BUILD VOCAB
    # =========================
    print("\nBuilding vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocab(df[CLEAN_TEXT_COL].tolist(), max_size=VOCAB_SIZE)

    # =========================
    # DATALOADER LSTM
    # =========================
    train_loader, val_loader, test_loader = get_lstm_dataloaders(
        df,
        vocab
    )

    # =========================
    # TRAIN BILSTM
    # =========================
    print("\nTraining BiLSTM...")
    model_lstm = BiLSTMClassifier(vocab_size=len(vocab.word2idx))

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
    # TRAIN BILSTM ATTENTION
    # =========================
    print("\nTraining BiLSTM + Attention...")
    model_att = BiLSTMAttentionClassifier(vocab_size=len(vocab.word2idx))

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
    # DATALOADER BERT
    # =========================
    print("\nPreparing BERT...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)

    train_bert, val_bert, test_bert = get_bert_dataloaders(
        df,
        tokenizer
    )

    # =========================
    # TRAIN DISTILBERT
    # =========================
    print("\nTraining DistilBERT...")
    model_bert = DistilBERTClassifier()

    train_model(
        model_bert,
        train_bert,
        val_bert,
        model_type="bert",
        save_path=os.path.join(DISTILBERT_MODEL_DIR, "model.pt"),
        epochs=BERT_EPOCHS,
        lr=BERT_LR,
        patience=BERT_PATIENCE,
    )

    # =========================
    # EVALUATION
    # =========================
    print("\nEvaluating...")

    criterion = torch.nn.CrossEntropyLoss()

    _, _, preds, labels = evaluate(model_lstm, test_loader, criterion)
    print("\nBiLSTM:")
    print_report(labels, preds)
    plot_confusion(labels, preds)

    _, _, preds, labels = evaluate(model_att, test_loader, criterion)
    print("\nBiLSTM + Attention:")
    print_report(labels, preds)
    plot_confusion(labels, preds)

    _, _, preds, labels = evaluate(model_bert, test_bert, criterion, True)
    print("\nDistilBERT:")
    print_report(labels, preds)
    plot_confusion(labels, preds)

    print("\n✅ DONE!")


# =========================
if __name__ == "__main__":
    main()