"""
download_data.py — Auto-Download Dataset IMDB dari Kaggle
========================================================
Mendownload dataset IMDB (50K reviews) dari Kaggle.
"""

import os
import shutil
import glob

from config import DATA_DIR, RAW_DATA_PATH, KAGGLE_DATASET, EXPECTED_FILENAME


def download_dataset() -> str:
    """
    Download dataset dari Kaggle ke folder Data/.

    Returns:
        str: Path ke file CSV
    """

    # ── Sudah ada? ──
    if os.path.exists(RAW_DATA_PATH):
        print(f"✅ Dataset sudah ada: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    os.makedirs(DATA_DIR, exist_ok=True)

    print("📥 Mendownload dataset dari Kaggle...")

    try:
        import kagglehub

        download_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"📦 Dataset didownload ke: {download_path}")

        # cari file CSV sesuai nama
        csv_files = glob.glob(
            os.path.join(download_path, "**", EXPECTED_FILENAME),
            recursive=True,
        )

        # fallback kalau nama beda
        if not csv_files:
            csv_files = glob.glob(
                os.path.join(download_path, "**", "*.csv"),
                recursive=True,
            )

        if csv_files:
            src = csv_files[0]
            shutil.copy2(src, RAW_DATA_PATH)
            print(f"✅ Dataset disalin ke: {RAW_DATA_PATH}")
            return RAW_DATA_PATH
        else:
            raise FileNotFoundError("CSV tidak ditemukan setelah download")

    except Exception as e:
        print(f"⚠️ kagglehub gagal: {e}")
        print("🔄 Mencoba fallback opendatasets...")
        return _fallback_opendatasets()


def _fallback_opendatasets() -> str:
    """Fallback pakai opendatasets"""
    try:
        import opendatasets as od

        kaggle_url = (
            "https://www.kaggle.com/datasets/"
            + KAGGLE_DATASET
        )

        od.download(kaggle_url, data_dir=DATA_DIR)

        downloaded = glob.glob(
            os.path.join(DATA_DIR, "**", "*.csv"),
            recursive=True
        )

        if downloaded:
            src = downloaded[0]
            shutil.copy2(src, RAW_DATA_PATH)
            print(f"✅ Dataset tersedia di: {RAW_DATA_PATH}")
            return RAW_DATA_PATH
        else:
            raise FileNotFoundError("CSV tidak ditemukan")

    except Exception as e:
        raise RuntimeError(
            f"Gagal download dataset: {e}\n"
            "Download manual dari:\n"
            "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews\n"
            f"Lalu taruh 'IMDB Dataset.csv' di: {DATA_DIR}"
        ) from e


if __name__ == "__main__":
    path = download_dataset()
    print(f"\n📄 Dataset siap: {path}")