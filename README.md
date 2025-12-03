# Wedding Recommendation System

Sistem rekomendasi pernikahan yang menggabungkan AI Intent Classification, Database MySQL, dan K-Means Clustering.

## Fitur Utama

1. **AI Intent Classification**: Memahami maksud pertanyaan user (mencari venue, catering, dekorasi, dll)
2. **Slot Extraction**: Mengekstrak informasi penting (tema, lokasi, budget, jumlah tamu)
3. **Database Query**: Mencari item yang sesuai dari database MySQL
4. **K-Means Clustering**: Mengelompokkan hasil dan merekomendasikan yang paling relevan

## Struktur File

```
transformers/
├── main.py                          # File utama untuk menjalankan sistem
├── conn.py                          # Modul koneksi database
├── recommendation.py                # Modul clustering & rekomendasi
├── local_transformer_intent.py     # Model AI untuk intent classification
├── requirements.txt                 # Dependencies Python
├── dataset_pertanyaan_wedding.csv  # Dataset training
└── models/
    └── local_transformer_intent/
        ├── model.pt                 # Model weights
        └── intent_labels.json       # Label intent
```

## Instalasi

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Setup Database

1. Buat database MySQL bernama `sepasangwp`
2. Import file `sepasangwp.sql` ke database:

```powershell
# Menggunakan MySQL CLI
mysql -u root -p sepasangwp < sepasangwp.sql

# Atau menggunakan phpMyAdmin (Import via web interface)
```

3. Konfigurasi koneksi database di `conn.py` (jika perlu):
   - Host: `localhost` (default)
   - Database: `sepasangwp`
   - User: `root` (default)
   - Password: `` (kosong, default)

### 3. Pastikan Model AI Sudah Dilatih

Jika model belum ada, latih terlebih dahulu:

```powershell
python local_transformer_intent.py --mode train --csv dataset_pertanyaan_wedding.csv --epochs 2
```

## Cara Menggunakan

### Mode Web (Recommended) 🌐

Jalankan aplikasi web dengan visualisasi clustering:

```powershell
python app.py
```

Kemudian buka browser dan akses: **http://localhost:5000**

**Fitur Web Interface:**
- ✅ Input query yang user-friendly
- ✅ Visualisasi clustering dengan scatter plot 2D
- ✅ Step-by-step process yang jelas
- ✅ Intent probability bars
- ✅ Rekomendasi dalam bentuk card yang menarik
- ✅ Real-time processing

### Mode Interaktif CLI

Jalankan sistem dengan mode interaktif:

```powershell
python main.py
```

Contoh interaksi:

```
> rekomendasi pernikahan adat sunda di jakarta budget berkisar paling mentok 300 juta dan paling kecil 200juta dan bisa untuk 500 tamu

STEP 1: AI INTENT CLASSIFICATION & SLOT EXTRACTION
📝 Input:
   rekomendasi pernikahan adat sunda di jakarta budget berkisar paling mentok 300 juta dan paling kecil 200juta dan bisa untuk 500 tamu

🤖 AI Prediction:
   Intent: cari_rekomendasi_paket
   Confidence: 59.76%

🎯 Extracted Slots:
   • tema: sunda
   • lokasi: jakarta
   • budget_min: Rp 200,000,000
   • budget_max: Rp 300,000,000
   • jumlah_tamu: 500

STEP 2: DATABASE QUERY
🔍 Query Parameters:
   • Intent: cari_rekomendasi_paket
   • Category Filter: All Categories
   • Tema: sunda
   • Lokasi: jakarta
   • Budget: Rp 200,000,000 - Rp 300,000,000

📦 Query Result: 15 items found

STEP 3: K-MEANS CLUSTERING & RECOMMENDATION
[Hasil clustering dan rekomendasi ditampilkan]
```

### Test Koneksi Database

```powershell
python conn.py
```

### Test Recommendation Engine

```powershell
python recommendation.py
```

## Alur Kerja Sistem

```
User Input (Text)
      ↓
[1] AI Intent Classification & Slot Extraction
    - Deteksi intent (cari_venue, cari_catering, dll)
    - Ekstrak slot (tema, lokasi, budget, jumlah_tamu)
      ↓
[2] Database Query
    - Query items dari database MySQL
    - Filter berdasarkan slot yang diekstrak
    - Menggunakan prepared statement untuk keamanan
      ↓
[3] K-Means Clustering
    - Prepare features (price, relevance, budget_fit)
    - Clustering items ke dalam N clusters
    - Hitung relevance score untuk setiap item
      ↓
[4] Recommendation
    - Pilih cluster dengan relevance tertinggi
    - Urutkan item dalam cluster berdasarkan score
    - Return top 10 recommendations
      ↓
Output (Rekomendasi Terbaik)
```

## Contoh Pertanyaan

- "rekomendasi pernikahan adat jawa di bandung budget 50 juta untuk 200 tamu"
- "cari venue outdoor di jakarta budget maksimal 100 juta"
- "paket catering untuk 500 orang budget 40-60 juta"
- "dekorasi pernikahan minimalis modern di bali"
- "MUA untuk pernikahan adat sunda di bandung"

## Troubleshooting

### Error: Tidak bisa terhubung ke database

- Pastikan MySQL/MariaDB sudah berjalan
- Cek username dan password di `conn.py`
- Pastikan database `sepasangwp` sudah dibuat dan diimport

### Error: Model tidak ditemukan

- Pastikan sudah melatih model dengan `python local_transformer_intent.py --mode train`
- Cek folder `models/local_transformer_intent/` sudah ada dan berisi `model.pt`

### Error: ModuleNotFoundError

- Install dependencies: `pip install -r requirements.txt`

## Teknologi yang Digunakan

- **Python 3.8+**
- **PyTorch**: Deep learning framework untuk AI model
- **Scikit-learn**: K-Means Clustering dan preprocessing
- **MySQL/MariaDB**: Database untuk menyimpan data items
- **mysql-connector-python**: Driver MySQL untuk Python

## Lisensi

Proyek ini dibuat untuk keperluan edukasi dan demonstrasi.
