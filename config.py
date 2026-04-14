"""
config.py — Konfigurasi Project IMDB Sentiment
==============================================
Menggunakan clean dataset hasil preprocessing.
"""

import os

# PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

RAW_CSV = os.path.join(DATA_DIR, "clean_imdb.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# KOLOM DATASET
TEXT_COL = "clean_review"
LABEL_COL = "sentiment"

# PYCARET SETTINGS
SESSION_ID = 42
TRAIN_SIZE = 0.8
N_TOP_MODELS = 3