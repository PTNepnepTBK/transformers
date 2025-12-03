# Update Log - Wedding Recommendation System

## Tanggal: 3 Desember 2025

### 🐛 Bug Fixes

1. **Fixed NoneType Error di recommendation.py**
   - Masalah: Error `'NoneType' object has no attribute 'lower'` ketika slot bernilai None
   - Solusi: Menambahkan validasi dan default value untuk semua slot
   - File: `recommendation.py` (line 53, 95)

### ✨ New Features

2. **Flexible Matching dengan Toleransi**
   - Implementasi toleransi 20% untuk budget (bisa lebih atau kurang)
   - Auto-relax filter jika tidak ada hasil yang ditemukan
   - Fallback mechanism: budget → lokasi → tema
   - File: `conn.py` (method `get_items_by_filter`)

3. **Web Interface dengan Flask**
   - UI modern dan responsive dengan gradient design
   - Real-time processing tanpa refresh halaman
   - File: `app.py`, `templates/index.html`

4. **Visualisasi Clustering**
   - Scatter plot 2D menggunakan PCA untuk dimensionality reduction
   - Highlight untuk top 5 recommendations dengan bintang emas
   - Warna berbeda untuk setiap cluster
   - Best cluster ditandai dengan border gold
   - File: `app.py` (function `create_cluster_visualization`)

### 📊 Improvements

5. **Enhanced Relevance Scoring**
   - Toleransi budget 20% untuk scoring
   - Bonus score jika tepat di range budget
   - Partial score untuk item di luar budget
   - File: `recommendation.py` (method `calculate_relevance_score`)

6. **Better Error Handling**
   - Semua None values dihandle dengan proper default
   - Try-catch untuk visualisasi
   - User-friendly error messages

### 📦 Dependencies Added

- Flask >= 2.3.0 (web framework)
- Matplotlib >= 3.7.0 (visualisasi)

## Cara Menggunakan

### Web Version (Recommended)

```powershell
# Install dependencies baru
pip install flask matplotlib

# Jalankan web app
python app.py
```

Akses di browser: **http://localhost:5000**

### CLI Version (Masih Tersedia)

```powershell
python main.py
```

## Contoh Query yang Dihandle dengan Baik

✅ **Query dengan data tidak lengkap:**
- "pernikahan lokasi bandung" → Akan cari semua items di Bandung
- "tema jawa di jakarta" → Akan cari tema jawa di Jakarta
- "venue budget 50 juta" → Toleransi 40-60 juta

✅ **Query dengan lokasi yang tidak ada:**
- "pernikahan tema sunda di jatim" → Fallback ke tema sunda saja
- "catering di bali untuk 1000 tamu" → Relax jumlah tamu jika perlu

✅ **Query dengan budget fleksibel:**
- "budget kisaran 50 juta" → Mencari 40-60 juta (±20%)
- "budget maksimal 100 juta" → Mencari hingga 120 juta

## Technical Details

### Flexible Matching Algorithm

```
1. Query database dengan filter awal + toleransi 20%
2. Jika hasil = 0:
   a. Relax budget constraint
   b. Jika masih 0, relax lokasi constraint
   c. Jika masih 0, relax tema constraint
3. Clustering pada hasil yang ditemukan
4. Ranking berdasarkan relevance score
```

### Clustering Visualization

```
1. Extract features: [price, tema_relevance, lokasi_relevance, budget_fit]
2. Normalize features dengan StandardScaler
3. K-Means clustering (default: 3 clusters)
4. PCA untuk reduce ke 2D untuk visualisasi
5. Plot dengan matplotlib:
   - Scatter plot untuk semua items
   - Warna berbeda per cluster
   - Gold stars untuk top 5 recommendations
```

## Screenshots

### Web Interface
- Modern gradient design (purple-pink)
- Step-by-step visualization
- Interactive cards
- Real-time clustering plot

### Clustering Visualization
- 2D scatter plot dengan PCA
- Color-coded clusters
- Gold stars untuk recommendations
- Legend dan grid

## Known Issues & Limitations

1. PCA dapat kehilangan informasi untuk visualisasi 2D
2. Visualisasi tidak muncul jika items < 2
3. K-Means fixed pada 3 clusters (bisa diubah di constructor)

## Future Improvements

- [ ] Dynamic number of clusters (auto-detect optimal K)
- [ ] Interactive cluster selection di web UI
- [ ] Export recommendations ke PDF
- [ ] Save search history
- [ ] User preferences untuk customize scoring
- [ ] 3D visualization option
