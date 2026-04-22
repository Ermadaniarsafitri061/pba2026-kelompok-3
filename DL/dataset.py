"""
dataset.py — Vocabulary, Dataset, dan DataLoader untuk IMDB
===========================================================
Siap untuk BiLSTM & DistilBERT (binary classification).
"""

import json
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    VOCAB_SIZE, MAX_LEN,
    LSTM_BATCH_SIZE,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
    VOCAB_PATH,
    CLEAN_DATA_PATH   # 
)


# ──────────────────────────────────────────────
# 📖 VOCABULARY
# ──────────────────────────────────────────────
class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX   = 0
    UNK_IDX   = 1

    def __init__(self):
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }

    def build_vocab(self, texts, max_size=VOCAB_SIZE):
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        most_common = counter.most_common(max_size - 2)
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"📖 Vocabulary: {len(self.word2idx):,} kata")

    def text_to_indices(self, text, max_len=MAX_LEN):
        tokens = text.split()[:max_len]
        indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        indices += [self.PAD_IDX] * (max_len - len(indices))
        return indices

    def save(self, path=VOCAB_PATH):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)
        print(f"💾 Vocabulary disimpan: {path}")


# ──────────────────────────────────────────────
# 📦 DATASET LSTM
# ──────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = self.vocab.text_to_indices(self.texts[idx], self.max_len)
        length = min(len(self.texts[idx].split()), self.max_len)
        length = max(length, 1)

        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )

# ──────────────────────────────────────────────
# 🔄 DATALOADER LSTM
# ──────────────────────────────────────────────
def get_lstm_dataloaders(df, vocab):

    texts = df["clean_review"].tolist()
    labels = df["label_encoded"].tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=(TEST_SIZE + VAL_SIZE),
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=(1 - relative_val),
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    train_ds = IMDBDataset(train_texts, train_labels, vocab)
    val_ds   = IMDBDataset(val_texts, val_labels, vocab)
    test_ds  = IMDBDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=LSTM_BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=LSTM_BATCH_SIZE)

    print(f"📦 Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader

# ──────────────────────────────────────────────
# 🚀 TEST CEPAT
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # ✅ FIX PATH DI SINI
    df = pd.read_csv(CLEAN_DATA_PATH)

    vocab = Vocabulary()
    vocab.build_vocab(df["clean_review"].tolist())

    train_loader, _, _ = get_lstm_dataloaders(df, vocab)

    for batch in train_loader:
        x, y, l = batch
        print("Shape:", x.shape, y.shape, l.shape)
        break