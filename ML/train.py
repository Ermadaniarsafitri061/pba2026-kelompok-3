"""
train.py — Training Pipeline IMDB (Fixed & Visualization Ready)
"""

import os
import warnings
import pandas as pd
import numpy as np

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
# SETUP PYCARET
# ══════════════════════════════════════════════
def setup_pycaret(df: pd.DataFrame):
    from pycaret.classification import setup

    print("Inisialisasi PyCaret...")

    df_model = df[[TEXT_COL, LABEL_COL]].copy()

    setup(
        data=df_model,
        target=LABEL_COL,
        text_features=[TEXT_COL],
        session_id=RANDOM_STATE,
        train_size=1 - TEST_SIZE,
        verbose=False,
        html=False,
        use_gpu=False,
        n_jobs=-1,  # 🔥 pakai semua CPU
    )

    print("Setup selesai.")


# ══════════════════════════════════════════════
# TRAIN MODELS
# ══════════════════════════════════════════════
def train_models():
    from pycaret.classification import create_model

    print("Training Logistic Regression, Naive Bayes, dan SVM...")

    models = {
        "Logistic Regression": create_model("lr"),
        "Naive Bayes": create_model("nb"),
        "SVM": create_model("svm"),
    }

    return models


# ══════════════════════════════════════════════
# FINALIZE MODEL
# ══════════════════════════════════════════════
def finalize_models(models):
    from pycaret.classification import finalize_model

    final_models = {}

    for name, model in models.items():
        print(f"Finalizing {name}...")
        final_models[name] = finalize_model(model)

    return final_models


# ══════════════════════════════════════════════
# SAVE MODEL
# ══════════════════════════════════════════════
def save_models(models):
    from pycaret.classification import save_model

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():
        path = os.path.join(MODEL_DIR, name.replace(" ", "_"))
        save_model(model, path)
        print(f"Model {name} disimpan di {path}.pkl")


# ══════════════════════════════════════════════
# 🔥 SAVE METRICS + PROBABILITIES
# ══════════════════════════════════════════════
def save_results(models):
    from pycaret.classification import predict_model, get_config
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print("Menyimpan hasil evaluasi...")

    os.makedirs("results", exist_ok=True)

    X_test = get_config("X_test")
    y_test = get_config("y_test")

    results_list = []

    # simpan y_test
    np.save("results/y_test.npy", y_test.values)

    for name, model in models.items():
        print(f"Processing {name}...")

        preds = predict_model(model, data=X_test)

        y_pred = preds["prediction_label"]

        # cari kolom probabilitas
        proba_col = None
        for col in preds.columns:
            if "score" in col.lower():
                proba_col = col
                break

        if proba_col is None:
            print(f"⚠️ {name} tidak punya probabilitas, skip PR curve")

        else:
            y_proba = preds[proba_col].values
            filename = name.replace(" ", "_").lower()
            np.save(f"results/y_proba_{filename}.npy", y_proba)

        y_proba = preds[proba_col].values

        # simpan per model
        filename = name.replace(" ", "_").lower()
        np.save(f"results/y_proba_{filename}.npy", y_proba)

        # 🔥 FIX LABEL (INI YANG PENTING)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="positive")
        rec = recall_score(y_test, y_pred, pos_label="positive")
        f1 = f1_score(y_test, y_pred, pos_label="positive")

        results_list.append([name, acc, prec, rec, f1])

    # simpan CSV
    df_results = pd.DataFrame(
        results_list,
        columns=["model", "accuracy", "precision", "recall", "f1"]
    )

    df_results.to_csv("results/results.csv", index=False)

    print("✅ Semua hasil disimpan di folder results/")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("Memuat dataset clean...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"Jumlah data: {len(df)}")

    # setup
    setup_pycaret(df)

    # train
    models = train_models()

    # finalize
    final_models = finalize_models(models)

    # save model
    save_models(final_models)

    # save hasil
    save_results(final_models)