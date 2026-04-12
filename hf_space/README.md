---
title: IMDB Sentiment Analysis
emoji: 🎬
colorFrom: yellow
colorTo: red
sdk: gradio
python_version: "3.10"
app_file: app.py
---

# IMDB Sentiment Analysis

Model klasifikasi sentimen (Positive / Negative) menggunakan PyCaret.

Dataset:
IMDB 50K Movie Reviews

Model:
- Logistic Regression / Naive Bayes / SVM (dipilih terbaik berdasarkan F1 Score)
- TF-IDF Vectorization via PyCaret

Cara penggunaan:
Masukkan review film pada textbox lalu klik submit untuk melihat hasil prediksi.