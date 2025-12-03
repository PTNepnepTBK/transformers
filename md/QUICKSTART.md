# 🚀 Quick Start Guide - Wedding Recommendation System

## Instalasi Dependencies

```powershell
# Install semua dependencies yang dibutuhkan
pip install flask matplotlib
```

Atau install semua sekaligus:

```powershell
pip install -r requirements.txt
```

## Cara Menjalankan

### 1️⃣ Web Interface (RECOMMENDED) 🌐

```powershell
python app.py
```

**Output yang diharapkan:**
```
🚀 Initializing Wedding Recommendation System...
   ✓ AI Model loaded
✓ Berhasil terhubung ke MySQL Server versi...
✓ Terhubung ke database: sepasangwp
   ✓ Database connected
   ✓ Recommendation Engine ready
✅ System ready!

================================================================================
🎊 WEDDING RECOMMENDATION SYSTEM - WEB VERSION 🎊
================================================================================

🌐 Server berjalan di: http://localhost:5000
📊 Akses statistik di: http://localhost:5000/api/stats

⌨️  Tekan Ctrl+C untuk menghentikan server
================================================================================
```

**Kemudian buka browser dan akses:**
- Main App: http://localhost:5000
- API Stats: http://localhost:5000/api/stats

### 2️⃣ CLI Interface (Alternative)

```powershell
python main.py
```

## Contoh Penggunaan

### Di Web Interface:

1. Buka http://localhost:5000
2. Ketik pertanyaan di search box
3. Klik tombol "🔍 Cari" atau tekan Enter
4. Lihat hasil step-by-step:
   - **Step 1**: AI memahami intent dan ekstrak slot
   - **Step 2**: Query database dengan filter
   - **Step 3**: Clustering dan visualisasi
   - **Step 4**: Top 10 rekomendasi

### Contoh Query yang Bagus:

✅ **Query Lengkap:**
```
rekomendasi pernikahan adat sunda di bandung budget 50-80 juta untuk 500 tamu
```

✅ **Query dengan Tema:**
```
dekorasi pernikahan tema minimalis modern di jakarta
```

✅ **Query dengan Budget:**
```
cari venue pernikahan budget maksimal 100 juta
```

✅ **Query Fleksibel:**
```
pernikahan adat jawa untuk 300 orang
```

✅ **Query Spesifik Kategori:**
```
MUA pengantin sunda di bandung
paket catering untuk 500 tamu
dokumentasi foto video pernikahan
```

## Troubleshooting

### ❌ Error: No module named 'flask'

**Solusi:**
```powershell
pip install flask matplotlib
```

### ❌ Error: Tidak bisa terhubung ke database

**Solusi:**
1. Pastikan MySQL/MariaDB sudah running
2. Pastikan database `sepasangwp` sudah dibuat
3. Import data dengan: `python import_data.py` (pilih opsi 3)

### ❌ Error: Model tidak ditemukan

**Solusi:**
```powershell
python local_transformer_intent.py --mode train
```

### ❌ Port 5000 sudah digunakan

**Solusi:**
Edit file `app.py` baris terakhir:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Ganti 5000 ke 5001
```

## Fitur Web Interface

### 🎨 Visual Design
- ✅ Modern gradient purple-pink theme
- ✅ Responsive untuk mobile & desktop
- ✅ Smooth animations
- ✅ Interactive hover effects

### 📊 Visualisasi Clustering
- ✅ 2D scatter plot dengan PCA
- ✅ Warna berbeda untuk setiap cluster
- ✅ Gold stars untuk top 5 recommendations
- ✅ Best cluster highlighted dengan border gold
- ✅ Legend dan grid

### 🔍 Real-time Processing
- ✅ Loading spinner saat processing
- ✅ Step-by-step visualization
- ✅ Intent probability bars
- ✅ Extracted slots display
- ✅ Cluster information cards
- ✅ Recommendation cards dengan hover effect

### 📋 Information Display
- ✅ Intent detection dengan confidence score
- ✅ All intent probabilities (bar chart)
- ✅ Extracted slots (tema, lokasi, budget, dll)
- ✅ Database query info (items found, category)
- ✅ Cluster statistics (size, avg price, avg relevance)
- ✅ Top 10 recommendations dengan detail lengkap

## API Endpoints

### POST /api/process
Request body:
```json
{
  "query": "rekomendasi pernikahan adat sunda di bandung"
}
```

Response:
```json
{
  "success": true,
  "query": "...",
  "ai_result": {
    "intent": "cari_rekomendasi_paket",
    "confidence": 0.5975,
    "slots": {...}
  },
  "database": {
    "total_items": 106
  },
  "clustering": {
    "n_clusters": 3,
    "clusters": [...]
  },
  "recommendations": [...],
  "visualization": "base64_encoded_image"
}
```

### GET /api/stats
Response:
```json
{
  "success": true,
  "total_items": 320,
  "categories": [...],
  "price_range": {
    "min_price": 50000,
    "max_price": 150000000
  }
}
```

## Tech Stack

- **Backend**: Python 3.12, Flask
- **AI Model**: PyTorch (Transformer Encoder)
- **Database**: MySQL/MariaDB
- **Clustering**: Scikit-learn (K-Means)
- **Visualization**: Matplotlib, PCA
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## Performance Tips

1. **Warm up**: Pertama kali jalankan bisa lambat (loading model)
2. **Caching**: Koneksi database di-cache
3. **Batch processing**: Clustering di-optimize untuk performa
4. **Visualization**: PCA sudah di-optimize untuk speed

## Next Steps

Setelah berhasil menjalankan:

1. ✅ Test dengan berbagai query
2. ✅ Lihat visualisasi clustering
3. ✅ Cek apakah recommendations relevan
4. ✅ Eksperimen dengan filter berbeda
5. ✅ Coba flexible matching (data tidak lengkap)

## Support

Jika ada masalah:
1. Cek error message di terminal
2. Cek browser console (F12)
3. Pastikan semua dependencies terinstall
4. Pastikan database terisi data

---

**Happy Wedding Planning! 🎊💍**
