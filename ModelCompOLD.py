import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, f1_score, precision_score, 
                             recall_score, confusion_matrix)

# ==========================================
# 1. pre-procesing (jika menggunakan data mentah)
# ==========================================

print("⏳ Memuat pustaka Sastrawi...")
stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text) 
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword_remover.remove(text)
    return text

# ==========================================
# 2. Load Dataset + injeksi kosakata toksik
# ==========================================

print("📂 Memuat dataset...")
try:
    df_train = pd.read_csv('Data_training.csv') # file data training
    df_train['label'] = pd.to_numeric(df_train['label'], errors='coerce').fillna(0).astype(int)
    print(f"   ✅ Data Training dimuat: {len(df_train)} baris")
except FileNotFoundError:
    print("   ❌ Error: File 'Data_training.csv' tidak ditemukan.")
    exit()

print("💉 Menyuntikkan kosakata toksik dengan Variasi Kalimat...") # injeksi hatespeech berdasarkan data uji (metode lexicon untuk imbalance data)
custom_hate_speech = [
    # Kata Tunggal
    "elek", "karbit", "kikir", "cupu", "taik", "mani",
    
    # Frasa Live Stream (Bigram)
    "l animasi", "game jelek", "l stream", "l juan",
    
    # Variasi Kalimat (Agar N-Gram menangkap konteks)
    "streamer cupu", "game ampas", "grafik elek", "dasar karbit", 
    "mani cok", "scam ini", "gameplay ampas", "l animasi l juan"
]

# injeksi x15
df_injection = pd.DataFrame({
    'text': custom_hate_speech * 15, # x15 frekuensi untuk memperkuat representasi
    'label': [1] * (len(custom_hate_speech) * 15) # x15 label 1 untuk setiap kalimat toksik
})
df_train = pd.concat([df_train, df_injection], ignore_index=True)

try:
    df_test = pd.read_csv('data_uji_clean.csv') # file data uji
    print(f"   ✅ Data Uji dimuat: {len(df_test)} baris")
except FileNotFoundError:
    print("   ❌ Error: File 'data_uji_livestreamL1.csv' tidak ditemukan.")
    exit()

is_labeled = df_test['label'].notna().any() and str(df_test['label'].iloc[0]).strip() != ''
if not is_labeled:
    print("\n⚠️ PERINGATAN: Kolom 'label' di data uji kosong. Sistem akan melakukan PREDIKSI SAJA.")

# ==========================================
# 3. run preprocesing
# ==========================================

print("⏳ Sedang melakukan text preprocessing...")
df_train['clean_text'] = df_train['text'].apply(preprocess_text)
df_test['clean_text'] = df_test['text'].apply(preprocess_text)

# ==========================================
# 4. Pembobotan TF-IDF with n-gram (Unigram + Bigram)
# ==========================================

print("🧮 Menghitung vektor TF-IDF (Unigram & Bigram)...")
start_tfidf = time.time()

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), # kombinasi Unigram (1 kata) dan Bigram (2 kata)
    max_features=5000 # batasi jumlah fitur tf-idf untuk efisiensi
)

X_train = vectorizer.fit_transform(df_train['clean_text']) 
y_train = df_train['label']
X_test = vectorizer.transform(df_test['clean_text'])

end_tfidf = time.time()
duration_tfidf = end_tfidf - start_tfidf

y_test = None
if is_labeled:
    y_test = pd.to_numeric(df_test['label'], errors='coerce').fillna(0).astype(int)


# ==========================================
# 5. Pelatihan & Uji Model
# ==========================================

print("\n🌲 Melatih Random Forest...")
start_rf = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, 
    # n_jobs=-1,
    #class_weight='balanced'
    )
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# thresholding manual (jika ingin menggunakan threshold khusus, misal 0.25)
# rf_probs = rf_model.predict_proba(X_test)[:, 1]
# rf_pred = (rf_probs >= 0.25).astype(int)

end_rf = time.time()
duration_rf = end_rf - start_rf
df_test['RF_Prediction'] = rf_pred


print("🚀 Melatih XGBoost...")
start_xgb = time.time()
xgb_model = XGBClassifier(
    eval_metric='logloss', 
    random_state=42,
    # n_jobs=-1,
    )
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# thresholding manual (jika ingin menggunakan threshold khusus, misal 0.25)
# xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
# xgb_pred = (xgb_probs >= 0.25).astype(int)

end_xgb = time.time()
duration_xgb = end_xgb - start_xgb
df_test['XGB_Prediction'] = xgb_pred 


# LOG WAKTU proses komputasi
print("\n" + "="*40)
print("⏱️  DURASI WAKTU PROSES KOMPUTASI")
print("="*40)
print(f"TF-IDF Vectorization : {duration_tfidf:.4f} detik")
print(f"Random Forest        : {duration_rf:.4f} detik")
print(f"XGBoost              : {duration_xgb:.4f} detik")
print("="*40)


# ==========================================
# 6. Evaluasi & Laporan Hasil
# ==========================================

if is_labeled and y_test is not None:
    print("\n" + "="*40)
    print("📊 HASIL EVALUASI (GROUND TRUTH TERSEDIA)")
    print("="*40)

    print("\n--- [1] Random Forest Report ---")
    print(classification_report(y_test, rf_pred, target_names=['Netral', 'Hate Speech']))
    
    print("\n--- [2] XGBoost Report ---")
    print(classification_report(y_test, xgb_pred, target_names=['Netral', 'Hate Speech']))

    # EKSTRAKSI METRIK KHUSUS KELAS HATE SPEECH (1) ---
    rf_prec = precision_score(y_test, rf_pred, pos_label=1)
    rf_rec = recall_score(y_test, rf_pred, pos_label=1)
    rf_f1 = f1_score(y_test, rf_pred, pos_label=1)

    xgb_prec = precision_score(y_test, xgb_pred, pos_label=1)
    xgb_rec = recall_score(y_test, xgb_pred, pos_label=1)
    xgb_f1 = f1_score(y_test, xgb_pred, pos_label=1)

    # VISUALISASI BAR CHART KOMPARASI
    print("\n📈 Membuat Grafik Perbandingan Metrik...")
    labels = ['Precision', 'Recall', 'F1-Score']
    rf_scores = np.array([rf_prec, rf_rec, rf_f1])
    xgb_scores = np.array([xgb_prec, xgb_rec, xgb_f1])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest', color='#2ca02c')
    rects2 = ax.bar(x + width/2, xgb_scores, width, label='XGBoost', color='#d62728')

    ax.set_ylabel('Skor (0.0 - 1.0)')
    ax.set_title('Komparasi Metrik Kelas Hate Speech\nRandom Forest vs XGBoost', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # label angka di atas bar
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    plt.tight_layout()
    plt.savefig('Grafik_Komparasi_HateSpeech.png', dpi=300)
    plt.show()

    # VISUALISASI CONFUSION MATRIX
    print("📉 Membuat Heatmap Confusion Matrix...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix RF
    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Netral', 'Hate Speech'], yticklabels=['Netral', 'Hate Speech'], ax=axes[0])
    axes[0].set_title('Confusion Matrix - Random Forest', fontweight='bold')
    axes[0].set_xlabel('Prediksi Model')
    axes[0].set_ylabel('Aktual (Ground Truth)')

    # Confusion Matrix XGBoost
    sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Netral', 'Hate Speech'], yticklabels=['Netral', 'Hate Speech'], ax=axes[1])
    axes[1].set_title('Confusion Matrix - XGBoost', fontweight='bold')
    axes[1].set_xlabel('Prediksi Model')
    axes[1].set_ylabel('Aktual (Ground Truth)')

    plt.tight_layout()
    plt.savefig('Grafik_Confusion_Matrix.png', dpi=300)
    plt.show()

    # Kesimpulan Akhir
    print("\n🏆 KESIMPULAN AKHIR:")
    if xgb_f1 > rf_f1:
        print(f"XGBoost lebih unggul {(xgb_f1 - rf_f1):.4f} poin pada F1-Score dibanding Random Forest.")
    elif rf_f1 > xgb_f1:
        print(f"Random Forest lebih unggul {(rf_f1 - xgb_f1):.4f} poin pada F1-Score dibanding XGBoost.")
    else:
        print("Kedua model memiliki F1-Score yang setara.")

else:
    print("\n" + "="*40)
    print("📝 HASIL PREDIKSI (DATA UJI BELUM DILABELI)")
    print("="*40)
    print("Menampilkan 5 prediksi pertama:")
    print(df_test[['timestamp', 'text', 'XGB_Prediction']].head())

# ==========================================
# 7. Simpan Hasil
# ==========================================
output_filename = 'hasil_klasifikasi_livestream.csv'
df_test.to_csv(output_filename, index=False)
print(f"\n💾 Hasil prediksi lengkap disimpan ke: {output_filename}")

print("🖼️ File grafik berhasil disimpan: 'Grafik_Komparasi_HateSpeech.png' & 'Grafik_Confusion_Matrix.png'")

