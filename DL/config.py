"""
config.py — Konfigurasi Project IMDB Sentiment Analysis
======================================================
Berisi path, parameter preprocessing, dan hyperparameter
untuk model Deep Learning (BiLSTM & DistilBERT).
"""

import os

# ──────────────────────────────────────────────
# 📁 PATH
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")

RAW_DATA_PATH   = os.path.join(DATA_DIR, "IMDB Dataset.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "clean_imdb_10k.csv")

# Buat folder kalau belum ada
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ──────────────────────────────────────────────
# 📋 DATASET
# ──────────────────────────────────────────────
TEXT_COL        = "review"
CLEAN_TEXT_COL  = "clean_review"
LABEL_COL       = "sentiment"

LABEL_MAPPING = {
    "negative": 0,
    "positive": 1
}

NUM_CLASSES = len(LABEL_MAPPING)

# ──────────────────────────────────────────────
# ⚙️ UMUM
# ──────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.2
VAL_SIZE    = 0.1

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# 🧠 HYPERPARAMETER — BiLSTM
# ──────────────────────────────────────────────
VOCAB_SIZE  = 20_000
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
MAX_LEN     = 128

LSTM_EPOCHS     = 10
LSTM_BATCH_SIZE = 64
LSTM_LR         = 1e-3
LSTM_PATIENCE   = 3

# ──────────────────────────────────────────────
# 🤗 HYPERPARAMETER — DistilBERT
# ──────────────────────────────────────────────
BERT_MODEL      = "distilbert-base-uncased"
BERT_MAX_LEN    = 128
BERT_EPOCHS     = 3
BERT_BATCH_SIZE = 16
BERT_LR         = 2e-5
BERT_PATIENCE   = 2

# ──────────────────────────────────────────────
# 💾 MODEL PATH
# ──────────────────────────────────────────────
BILSTM_MODEL_PATH     = os.path.join(MODEL_DIR, "bilstm.pt")
BILSTM_ATT_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_attention.pt")
DISTILBERT_MODEL_DIR  = os.path.join(MODEL_DIR, "distilbert")

VOCAB_PATH         = os.path.join(MODEL_DIR, "vocab.json")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.json")