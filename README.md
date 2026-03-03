# Analisis Komparasi XGBoost vs Random Forest: Klasifikasi Ujaran Kebencian pada Komentar Hasil Live Streaming YouTube

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
```

## ⚙️ Cara Penggunaan
1. Tahap Pengambilan Data (jika tidak ada data uji atau mencari data uji baru)
Buka YoutubechatSCRAPplus.py, masukkan URL video YouTube target (dibagian bawah kode program), lalu run file:
```bash
python YoutubechatSCRAPplus.py
```
Data akan tersimpan secara otomatis dalam format data_uji_clean.csv.

2. Tahap Klasifikasi & Komparasi
Jalankan skrip utama untuk melihat hasil analisis:
```bash
python ModelCompOLD.py
```
Program akan menampilkan laporan klasifikasi (Precision, Recall, F1-Score), conffusion matrix, dan durasi waktu proses (terminal). program juga akan menyimpan visualisasi grafik dan matrix dalam format .png, serta hasil klasifikasi pada data uji dalam format .csv.


## 📚 Referensi Utama
Dey, R. K., Sarddar, D., Bose, R., Sutradhar, S., Sarkar, I., & Roy, S. (2024). Combating Hate Speech on Social Media: Automated Detection using XGBoost Algorithm. Journal of Informatics Electrical and Electronics Engineering (JIEEE), 5(1), 1-10.

Jiang, H., He, Z., Ye, G., & Zhang, H. (2020). Network Intrusion Detection Based on PSO-Xgboost Model. IEEE Access, 8, 58392-58401.

Pamungkas, E. W., Purworini, D., Putri, D. G. P., & Akhtar, S. (2024). Enhancing hate speech detection in Indonesian using abusive words lexicon. Indonesian Journal of Electrical Engineering and Computer Science, 33(1), 450-462.

Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks, 106, 249-259.

Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. EMNLP-IJCNLP.
