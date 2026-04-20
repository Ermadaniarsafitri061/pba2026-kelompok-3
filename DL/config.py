"""
config.py — Konfigurasi & Konstanta untuk IMDB Sentiment Analysis
================================================================
Digunakan untuk klasifikasi sentimen (2 kelas: Positive / Negative)
"""

import os
import torch

# ──────────────────────────────────────────────
# 📁 PATH
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")

# 🔥 SESUAIKAN DENGAN FILE DATASET KAMU
RAW_CSV = os.path.join(DATA_DIR, "clean_imdb.csv")

# Buat folder kalau belum ada
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ──────────────────────────────────────────────
# 📋 DATASET (IMDB)
# ──────────────────────────────────────────────
TEXT_COL  = "review"      # ganti kalau nama kolom beda
LABEL_COL = "sentiment"   # ganti kalau nama kolom beda

# Label untuk sentiment analysis
LABEL_LIST = ["Negative", "Positive"]
NUM_CLASSES = len(LABEL_LIST)

# Mapping label (optional tapi bagus)
label2id = {
    "Negative": 0,
    "Positive": 1
}

id2label = {
    0: "Negative",
    1: "Positive"
}

# ──────────────────────────────────────────────
# ⚙️  UMUM
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# Gunakan None untuk pakai semua data
SAMPLE_SIZE = None  

TEST_SIZE   = 0.10
VAL_SIZE    = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# 🧠 HYPERPARAMETER — LSTM (OPTIONAL)
# ──────────────────────────────────────────────
VOCAB_SIZE  = 20_000
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
MAX_LEN     = 128

LSTM_EPOCHS     = 5
LSTM_BATCH_SIZE = 64
LSTM_LR         = 1e-3
LSTM_PATIENCE   = 2

# ──────────────────────────────────────────────
# 🤗 HYPERPARAMETER — DistilBERT (RECOMMENDED)
# ──────────────────────────────────────────────
BERT_MODEL      = "distilbert-base-uncased"
BERT_MAX_LEN    = 128
BERT_EPOCHS     = 3
BERT_BATCH_SIZE = 16
BERT_LR         = 2e-5
BERT_PATIENCE   = 2

# ──────────────────────────────────────────────
# 💾 PATH MODEL
# ──────────────────────────────────────────────
DISTILBERT_MODEL_DIR  = os.path.join(MODEL_DIR, "distilbert")
DISTILBERT_MODEL_PATH = os.path.join(DISTILBERT_MODEL_DIR, "distilbert.pt")

# Optional (kalau pakai LSTM)
BILSTM_MODEL_PATH     = os.path.join(MODEL_DIR, "bilstm.pt")
BILSTM_ATT_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_attention.pt")

# Optional tambahan
VOCAB_PATH         = os.path.join(MODEL_DIR, "vocab.json")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.json")