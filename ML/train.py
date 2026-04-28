"""
train.py — Training Pipeline IMDB (Clean Dataset)
==================================================
Menggunakan clean_imdb.csv tanpa preprocessing ulang.
"""

import os
import warnings
import pandas as pd

from config import (
    DATA_PATH,
    TEXT_COL,
    LABEL_COL,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
#  INI SETUP PYCARET
# ══════════════════════════════════════════════

def setup_pycaret(df: pd.DataFrame):
    from pycaret.classification import setup

    print("Inisialisasi PyCaret...")

    df_model = df[[TEXT_COL, LABEL_COL]].copy()

    setup(
        data=df_model,
        target=LABEL_COL,
        text_features=[TEXT_COL],
        session_id=RANDOM_STATE,   # ✅ FIX
        train_size=1 - TEST_SIZE,  # ✅ FIX
        verbose=True,
        html=False,
        use_gpu=False,
    )

    print("Setup selesai.")


# ══════════════════════════════════════════════
# COMPARE MODELS
# ══════════════════════════════════════════════

def compare_models_3():
    from pycaret.classification import compare_models

    print("Membandingkan Logistic Regression, Naive Bayes, dan SVM...")

    best = compare_models(
        include=["lr", "nb", "svm"],
        sort="F1",
        cross_validation=False,
    )

    print("Perbandingan selesai.")
    return best


# ══════════════════════════════════════════════
# FIKSASI MODEL
# ══════════════════════════════════════════════

def finalize_and_save(model):
    from pycaret.classification import finalize_model, save_model

    print("Finalisasi model...")
    final = finalize_model(model)

    save_path = os.path.join(MODEL_DIR, "sentiment_model")
    save_model(final, save_path)

    print(f"Model disimpan di: {save_path}.pkl")
    return final


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("Memuat dataset clean...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"Jumlah data: {len(df)}")
    print(df.head())

    setup_pycaret(df)

    best = compare_models_3()

    finalize_and_save(best)