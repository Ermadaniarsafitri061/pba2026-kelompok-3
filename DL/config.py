"""
config.py — Konfigurasi Project IMDB Sentiment Analysis
======================================================
Berisi path, hyperparameter model, dan informasi dataset
(IMDB dari Kaggle, diambil 10.000 sample).
"""

import os
import torch

# ──────────────────────────────────────────────
# 📁 PATH
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")

RAW_DATA_PATH = os.path.join(
    DATA_DIR,
    "imdb-dataset-of-50k-movie-reviews",
    "IMDB Dataset.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "IMDB Dataset.csv")

# Buat folder kalau belum ada
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ──────────────────────────────────────────────
# 📋 DATASET
# ──────────────────────────────────────────────
# Dataset: IMDB Movie Reviews (50K)
# Sumber: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# Digunakan: 10.000 sample (stratified)

KAGGLE_DATASET    = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
EXPECTED_FILENAME = "IMDB Dataset.csv"

TEXT_COL       = "review"
CLEAN_TEXT_COL = "clean_review"
LABEL_COL      = "sentiment"

LABEL_MAPPING = {
    "negative": 0,
    "positive": 1
}

LABEL_LIST = ["negative", "positive"]
NUM_CLASSES = len(LABEL_MAPPING)

# ──────────────────────────────────────────────
# ⚙️ UMUM
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# jumlah data yang diambil dari raw dataset
SAMPLE_SIZE = 10_000

TEST_SIZE = 0.2
VAL_SIZE  = 0.1

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