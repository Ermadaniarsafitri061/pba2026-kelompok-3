---

title: IMDB Sentiment Analyzer (Deep Learning)
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# 🎬 IMDB Sentiment Analyzer

Aplikasi klasifikasi sentimen teks menggunakan model Deep Learning berbasis PyTorch.

---

## 📌 Deskripsi

Aplikasi ini digunakan untuk menganalisis teks (review film) dan mengklasifikasikannya menjadi dua kategori sentimen:

* **Positive** — ulasan positif
* **Negative** — ulasan negatif

Model dilatih menggunakan dataset IMDB yang berisi review film berbahasa Inggris.

---

## 🤖 Model yang Digunakan

| Model          | Deskripsi                                                                                          |
| -------------- | -------------------------------------------------------------------------------------------------- |
| **DistilBERT** | Model transformer ringan hasil fine-tuning, memberikan performa terbaik untuk klasifikasi sentimen |

> Model Deep Learning dikembangkan menggunakan **PyTorch** dan **Hugging Face Transformers**

---

## 🚀 Cara Menggunakan

1. Masukkan teks review film (bahasa Inggris)
2. Klik tombol **Analisis**
3. Sistem akan menampilkan:

   * Prediksi sentimen (Positive / Negative)
   * Confidence score untuk masing-masing kelas

---

## 📊 Dataset

Dataset yang digunakan:

**IMDB Movie Reviews Dataset**

Karakteristik:

* Bahasa: Inggris
* Task: Sentiment Classification
* Label:

  * 0 → Negative
  * 1 → Positive
* Jumlah data: ±50.000 review

---

## ⚙️ Teknologi yang Digunakan

* PyTorch
* Hugging Face Transformers
* Gradio
* Scikit-learn

---

## ⚠️ Disclaimer

> Aplikasi ini dibuat untuk keperluan akademis dalam mata kuliah **Pemrosesan Bahasa Alami (PBA)** di Institut Teknologi Sumatera (ITERA).
> Model mungkin tidak selalu akurat dan tidak dimaksudkan untuk penggunaan produksi.

---

## 👨‍💻 Author

*Workshop PBA — Institut Teknologi Sumatera (ITERA), 2026*

---
