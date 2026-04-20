"""
config.py — Konfigurasi Project IMDB Sentiment
"""

import os

# ROOT PROJECT
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FOLDER
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# DATASET 
DATA_PATH = os.path.join(DATA_DIR, "clean_imdb.csv")

# pastikan folder model ada
os.makedirs(MODEL_DIR, exist_ok=True)

# KOLOM DATASET
TEXT_COL = "clean_review"
LABEL_COL = "sentiment"

# TRAINING CONFIG
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000