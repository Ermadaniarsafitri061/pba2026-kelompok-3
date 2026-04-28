"""
config.py — Konfigurasi Project IMDB Sentiment
"""

import os

# =========================
# ROOT PROJECT (NAIK 1 LEVEL DARI ML)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# FOLDER
# =========================
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =========================
# DATASET
# =========================
DATA_PATH = os.path.join(DATA_DIR, "clean_imdb.csv")

# pastikan folder model tersedia
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# KOLOM DATASET
# =========================
TEXT_COL = "clean_review"
LABEL_COL = "sentiment"

# =========================
# TRAINING CONFIG
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000


# =========================
# DEBUG (opsional, tapi bagus)
# =========================
if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("DATA_PATH:", DATA_PATH)
    print("File exists?", os.path.exists(DATA_PATH))