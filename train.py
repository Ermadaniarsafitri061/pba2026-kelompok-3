"""
train.py — Training Pipeline IMDB (Clean Dataset)
==================================================
Menggunakan clean_imdb.csv tanpa download & preprocess ulang.
"""

import os
import warnings
import pandas as pd

from config import (
    RAW_CSV,
    LABEL_COL,
    MODEL_DIR,
    SESSION_ID,
    TRAIN_SIZE,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
# SETUP PYCARET
# ══════════════════════════════════════════════

def setup_pycaret(df: pd.DataFrame):
    from pycaret.classification import setup

    print("Inisialisasi PyCaret...")

    df_model = df[["clean_review", LABEL_COL]].copy()

    s = setup(
        data=df_model,
        target=LABEL_COL,
        text_features=["clean_review"],
        session_id=SESSION_ID,
        train_size=TRAIN_SIZE,
        verbose=True,
        html=False,
        use_gpu=False,
    )

    print("Setup selesai.")
    return s


# ══════════════════════════════════════════════
# COMPARE 3 MODELS
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
# FINALISASI MODEL 
# ══════════════════════════════════════════════

def finalize_and_save(model):
    from pycaret.classification import finalize_model, save_model

    print("Finalisasi model...")
    final = finalize_model(model)

    save_path = os.path.join(MODEL_DIR, "best_model")
    save_model(final, save_path)

    print(f"Model disimpan di: {save_path}.pkl")
    return final


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("Memuat dataset clean...")
    df = pd.read_csv(RAW_CSV)

    setup_pycaret(df)

    best = compare_models_3()

    finalize_and_save(best)