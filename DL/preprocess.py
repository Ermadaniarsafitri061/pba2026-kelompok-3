"""
preprocess.py — Cleaning IMDB Dataset (10K Sample)
=================================================
Membersihkan teks review dan encode label (binary),
dengan sampling 10.000 data (stratified).
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    TEXT_COL,
    LABEL_COL,
    LABEL_MAPPING,
)


# ──────────────────────────────────────────────
# 🧹 CLEAN TEXT
# ──────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML
    text = re.sub(r"[^a-z\s]", " ", text)                # non-alpha
    text = re.sub(r"\s+", " ", text).strip()             # spasi

    return text


# ──────────────────────────────────────────────
# 🚀 MAIN PREPROCESS
# ──────────────────────────────────────────────
def preprocess():
    print("📥 Load dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    df = df[[TEXT_COL, LABEL_COL]].copy()
    df.dropna(inplace=True)

    print(f"📊 Jumlah data awal: {len(df):,}")

    # ── 🔥 SAMPLING 10K (STRATIFIED) ──
    df, _ = train_test_split(
        df,
        train_size=10000,
        stratify=df[LABEL_COL],
        random_state=42
    )

    df = df.reset_index(drop=True)

    print(f"🔀 Setelah sampling: {len(df):,}")

    # ── CLEANING ──
    print("🧹 Cleaning text...")
    df["clean_review"] = df[TEXT_COL].apply(clean_text)

    df = df[df["clean_review"].str.len() > 0]

    # ── ENCODE LABEL ──
    print("🔢 Encoding label...")
    df["label_encoded"] = df[LABEL_COL].map(LABEL_MAPPING)

    print(f"📊 Setelah cleaning: {len(df):,}")

    # ── SIMPAN ──
    df.to_csv(CLEAN_DATA_PATH, index=False)

    print(f"✅ Data bersih disimpan di: {CLEAN_DATA_PATH}")

    return df


# ──────────────────────────────────────────────
# 🔍 SAMPLE OUTPUT
# ──────────────────────────────────────────────
def show_sample(df, n=5):
    print("\n=== SAMPLE HASIL CLEANING ===")
    for _, row in df.sample(n=n, random_state=42).iterrows():
        print(f"\nLabel: {row[LABEL_COL]}")
        print(f"Asli : {row[TEXT_COL][:100]}")
        print(f"Bersih: {row['clean_review'][:100]}")


if __name__ == "__main__":
    df = preprocess()
    show_sample(df)
