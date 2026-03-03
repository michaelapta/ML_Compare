# Analisis Komparasi XGBoost vs Random Forest: Klasifikasi Ujaran Kebencian pada YouTube (Netizen +62)

Proyek ini merupakan implementasi penelitian skripsi yang berfokus pada deteksi otomatis ujaran kebencian (*hate speech*) pada kolom komentar hasil live streaming (VOD) melalui platform YouTube. Penelitian ini menggunakan pendekatan teknik **Lexicon Injection** untuk meningkatkan akurasi deteksi pada domain spesifik ujaran kebencian data uji (Gaming/Live Streaming).

## 🚀 Fitur Utama
* **Hybrid YouTube Scraper:** Pengambilan data komentar hasil live streaming (VOD).
* **Indonesian NLP Pipeline:** Pemrosesan teks menggunakan pustaka Sastrawi (Stemming & Stopword Removal).
* **Lexicon Injection (15x):** Teknik augmentasi data manual untuk memperkuat bobot TF-IDF pada kata-kata kunci toksik spesifik (Gaming Slang).
* **Performance Analytics:** Komparasi akurasi (F1-Score) dan efisiensi waktu komputasi secara presisi dengan persyaratan menggunakan 'Base Model' dari kedua model yang akan dikomparasi.

---

## 🛠️ Prasyarat & Instalasi

Pastikan Anda telah menginstal Python 3.8 ke atas. Untuk menginstal semua *library* yang dibutuhkan, jalankan perintah berikut:

```bash
pip install pandas numpy scikit-learn xgboost Sastrawi matplotlib seaborn youtube-comment-downloader tqdm

