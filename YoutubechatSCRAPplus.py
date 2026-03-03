import pandas as pd
import re
import time
from urllib.parse import parse_qs, urlparse

# Pustaka Sastrawi untuk Preprocessing
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Pustaka Scraping & Progress Bar
from youtube_comment_downloader import SORT_BY_RECENT, YoutubeCommentDownloader
from tqdm import tqdm

# ==========================================
# 1. INISIALISASI PREPROCESSING SASTRAWI
# ==========================================
print("⏳ Memuat pustaka Sastrawi (Stopword)...")
stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text) 
    text = re.sub(r'[^a-z\s]', ' ', text)      
    text = re.sub(r'\s+', ' ', text).strip()   
    text = stopword_remover.remove(text)
    return text

tqdm.pandas(desc="🧹 Membersihkan Teks")

# ==========================================
# 2. FUNGSI SCRAPING YOUTUBE
# ==========================================
def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in {"youtu.be"}: return parsed.path.strip("/")
    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query and query["v"]: return query["v"][0]
        match = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", parsed.path)
        if match: return match.group(1)
        match = re.search(r"/live/([A-Za-z0-9_-]{6,})", parsed.path)
        if match: return match.group(1)
    raise ValueError("URL YouTube tidak valid.")

def scrape_comments(url: str, max_items: int = 100) -> list[dict]:
    downloader = YoutubeCommentDownloader()
    generator = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
    comments = []
    
    with tqdm(total=max_items, desc="📥 Scraping Komentar Biasa", unit=" chat") as pbar:
        for item in generator:
            text, timestamp = str(item.get("text", "")).strip(), str(item.get("time", "")).strip()
            if text: 
                comments.append({"timestamp": timestamp, "text": text})
                pbar.update(1) 
            if len(comments) >= max_items: break
    return comments

# PERBAIKAN: Mengubah tipe start_time menjadi Integer (Detik)
def scrape_live_chat(url: str, max_items: int = 100, start_time: int = 0) -> list[dict]:
    try: import pytchat
    except Exception as err: raise RuntimeError("Library pytchat belum terpasang.") from err

    video_id = extract_video_id(url)
    chat = None
    try: chat = pytchat.create(video_id=video_id, seektime=start_time)
    except Exception as err: raise ValueError(f"Gagal akses live chat (Video ID ditolak): {err}") from err

    try:
        # Pengecekan awal, jika video tidak punya live chat, langsung putus
        if not chat.is_alive(): 
            raise ValueError("Koneksi ditolak oleh YouTube. (Alasan: Video ini TIDAK memiliki Live Chat Replay, atau dibatasi oleh server).")

        comments = []
        empty_round = 0
        max_empty_rounds = 10 
        attempt_count = 0
        max_attempts = 150 

        with tqdm(total=max_items, desc="📥 Scraping Live Chat", unit=" chat") as pbar:
            while len(comments) < max_items and attempt_count < max_attempts:
                attempt_count += 1
                try:
                    if not chat.is_alive(): break
                    data = chat.get()
                    
                    if not data:
                        empty_round += 1
                        if empty_round >= max_empty_rounds: break
                        time.sleep(1)
                        continue
                    
                    if hasattr(data, 'sync_items'): items = data.sync_items() # type: ignore
                    elif hasattr(data, 'items'): items = data.items # type: ignore
                    elif isinstance(data, (list, tuple)): items = data
                    else: items = []

                    if not items:
                        empty_round += 1
                        if empty_round >= max_empty_rounds: break
                        time.sleep(1)
                        continue

                    empty_round = 0
                    for item in items:
                        message = str(getattr(item, "message", "")).strip()
                        timestamp = str(getattr(item, "elapsedTime", "")).strip() 
                        
                        if timestamp.startswith("-"): continue
                        
                        if message and len(message) > 0:
                            comments.append({"timestamp": timestamp, "text": message})
                            pbar.update(1) 
                        if len(comments) >= max_items:
                            return comments
                except Exception:
                    empty_round += 1
                    if empty_round >= max_empty_rounds: break
                    time.sleep(1) 
                    continue

        if not comments: raise ValueError("Gagal total. Tidak ada chat di menit ini.")
        return comments
    finally:
        if chat:
            try: chat.terminate()
            except: pass

# ==========================================
# 3. EKSEKUSI PROGRAM UTAMA
# ==========================================
if __name__ == "__main__":
    
    URL_TARGET = "https://www.youtube.com/live/FUcY259bPuE?si=xeMiu0MkmeZIukhM" 
    JUMLAH_DATA = 200 
    
    # PERBAIKAN: Gunakan DETIK MURNI (Integer). 300 = Menit ke 5:00.
    WAKTU_MULAI_DETIK = 3519 
    
    print(f"\n🚀 Memulai ekstraksi dari URL: {URL_TARGET}")
    print(f"⏱️ Mulai dari detik ke-: {WAKTU_MULAI_DETIK}")
    print("-" * 50)
    
    try:
        # smart mode ("auto")
        # try Live Chat first, jika gagal akan ambil Komentar Reguler
        try:
            source = "Live Chat"
            scraped_data = scrape_live_chat(url=URL_TARGET, max_items=JUMLAH_DATA, start_time=WAKTU_MULAI_DETIK)
        except ValueError as err_live:
            print(f"⚠️ {err_live}")
            print("🔄 Banting setir otomatis: Mengambil Komentar Biasa di bawah video...\n")
            source = "Komentar Biasa"
            scraped_data = scrape_comments(url=URL_TARGET, max_items=JUMLAH_DATA)
            
        print(f"\n✅ Berhasil mengambil {len(scraped_data)} data dari {source}!\n")
        
        # PROSES PANDAS
        df_raw = pd.DataFrame(scraped_data)
        df_raw['label'] = "" 
        df_raw = df_raw[['timestamp', 'text', 'label']] 
        
        df_clean = df_raw.copy()
        df_clean['text'] = df_clean['text'].progress_apply(preprocess_text) 
        
        file_raw = "data_uji_raw.csv"
        file_clean = "data_uji_clean.csv"
        df_raw.to_csv(file_raw, index=False, encoding='utf-8')
        df_clean.to_csv(file_clean, index=False, encoding='utf-8')
        
        print("\n" + "=" * 50)
        print("🎉 PROSES SELESAI!")
        print(f"📂 File mentah   : {file_raw}")
        print(f"✨ File bersih   : {file_clean}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Terjadi kesalahan sistem: {e}")