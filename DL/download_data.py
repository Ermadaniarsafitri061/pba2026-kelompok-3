"""
download_data.py — Load Dataset IMDB (Manual)
=============================================
Digunakan untuk memastikan dataset tersedia di folder data/.
"""

import os
from config import RAW_DATA_PATH


def load_dataset_path():
    """
    Cek apakah dataset IMDB sudah tersedia.

    Returns:
        str: path ke dataset
    """
    if os.path.exists(RAW_DATA_PATH):
        print(f"✅ Dataset ditemukan: {RAW_DATA_PATH}")
        return RAW_DATA_PATH
    else:
        raise FileNotFoundError(
            f"❌ Dataset tidak ditemukan!\n"
            f"Silakan letakkan file 'IMDB Dataset.csv' ke folder data/"
        )


if __name__ == "__main__":
    path = load_dataset_path()
    print(f"\n📄 Dataset siap digunakan: {path}")
