# Wedding Recommendation System - Update Summary

## 🎊 New Features

### 1. Wedding Package Planner (Paket Pernikahan Lengkap)

Sistem sekarang dapat **merancang paket pernikahan lengkap** yang mencakup semua kategori vendor yang dibutuhkan:

#### Kategori Esensial:
- ✅ Venue (Tempat pernikahan)
- ✅ Catering (Makanan & minuman)
- ✅ Decoration (Dekorasi)
- ✅ MUA (Make Up Artist)
- ✅ Docummentation (Foto & Video)

#### Kategori Opsional:
- ✅ Attire (Busana pengantin)
- ✅ Entertainment (Hiburan - band, musik)
- ✅ Master of Ceremony (MC)
- ✅ Traditional Ceremony (Upacara adat)

### 2. Database Expansion

Database diperluas dengan **345 items baru** untuk 3 tema utama:

#### Tema Sunda (115 items):
- Lokasi: Bandung, Bogor, Sukabumi, Cirebon, Tasikmalaya, Garut
- Budaya: Ngeuyeuk seureuh, siger, kebaya sunda, gamelan degung

#### Tema Jawa (115 items):
- Lokasi: Yogyakarta, Solo, Semarang, Surabaya, Malang, Magelang
- Budaya: Siraman, midodareni, paes, gamelan jawa

#### Tema Betawi (115 items):
- Lokasi: Jakarta (5 wilayah), Tangerang
- Budaya: Buka palang pintu, none jakarta, gambang kromong

**Total database sekarang: 680+ items**

---

## 🚀 Cara Menggunakan

### A. Generate Data Baru

```bash
python expand_database.py
```

Output:
- `generated_data_sunda.sql` / `.json`
- `generated_data_jawa.sql` / `.json`
- `generated_data_betawi.sql` / `.json`
- `generated_data_all.sql` / `.json`

### B. Import Data ke Database

```bash
python import_data.py generated_data_all.sql
```

### C. Test Package Planner (CLI)

```bash
python package_planner.py
```

### D. Run Web Application

```bash
python app.py
```

Buka: `http://localhost:5000`

---

## 📝 Contoh Query Paket Pernikahan

### Query yang otomatis dideteksi sebagai **Paket Lengkap**:

```
✅ rekomendasi pernikahan adat sunda di bandung
✅ paket pernikahan lengkap adat jawa di yogyakarta
✅ rancang pernikahan adat betawi di jakarta budget 200 juta
✅ acara pernikahan adat sunda di bandung untuk 500 tamu
✅ planning pernikahan tema jawa di solo budget 150-200 juta
✅ estimasi budget pernikahan adat sunda di bandung
```

### Query untuk kategori spesifik (tidak paket):

```
❌ cari venue pernikahan di bandung
❌ MUA pengantin sunda di bandung
❌ dokumentasi foto video pernikahan
❌ sewa kebaya sunda untuk pengantin
```

---

## 🎯 Fitur Package Planner

### 1. Multi-Category Search
Sistem akan mencari items dari **semua 9 kategori** sekaligus berdasarkan:
- ✅ Tema (sunda/jawa/betawi)
- ✅ Lokasi
- ✅ Budget range
- ✅ Jumlah tamu

### 2. Three Package Tiers

System otomatis membuat **3 pilihan paket**:

#### 💚 Paket BUDGET
- Item termurah dari setiap kategori
- Total harga minimum
- Cocok untuk budget terbatas

#### 💙 Paket STANDARD
- Item mid-range dari setiap kategori
- Balance antara harga dan kualitas
- Pilihan paling umum

#### 💛 Paket PREMIUM
- Item termahal dari setiap kategori
- Total harga maksimum
- Untuk yang menginginkan yang terbaik

### 3. Price Breakdown

```
💰 ESTIMASI BIAYA PAKET:
   • Paket Minimum: Rp 102,500,000
   • Paket Maksimum: Rp 112,000,000
   • Paket Rata-rata: Rp 105,666,666

📦 BREAKDOWN PER KATEGORI:
   Venue: Rp 35,000,000
   Catering: Rp 14,000,000
   Decoration: Rp 25,000,000
   MUA: Rp 4,000,000
   Documentation: Rp 10,000,000
   ...
```

### 4. Budget Status

System akan membandingkan dengan budget user:

- ✅ **Fully Affordable**: Semua pilihan dalam budget
- ⚠️ **Partially Affordable**: Beberapa pilihan dalam budget
- ❌ **Over Budget**: Paket melebihi budget

### 5. Smart Detection

System otomatis menambahkan kategori opsional berdasarkan konteks:

```python
Query: "pernikahan adat sunda" 
→ Otomatis menambahkan "Traditional Ceremony" (upacara adat)

Query: "pernikahan" (tanpa adat)
→ Hanya kategori esensial
```

---

## 📊 Statistics

### Database Content:

```
Total Items: 680+

By Category:
- Venue: 45 items
- Catering: 45 items
- Decoration: 45 items
- MUA: 45 items
- Documentation: 45 items
- Attire: 30 items
- Entertainment: 30 items
- Master of Ceremony: 30 items
- Traditional Ceremony: 30 items

By Theme:
- Sunda: 115 items
- Jawa: 115 items
- Betawi: 115 items
- Others: 335+ items
```

---

## 🔧 Technical Details

### New Files Created:

1. **`package_planner.py`**
   - Class: `WeddingPackagePlanner`
   - Methods: 
     - `detect_package_request()` - Deteksi query paket
     - `create_package_recommendations()` - Generate paket lengkap
     - `get_category_items()` - Ambil items per kategori
     - `calculate_package_price_ranges()` - Hitung range harga
     - `_create_package_tiers()` - Buat 3 tier paket

2. **`expand_database.py`**
   - Class: `WeddingDataGenerator`
   - Methods untuk generate data per kategori:
     - `generate_venues()`
     - `generate_catering()`
     - `generate_decoration()`
     - `generate_mua()`
     - `generate_documentation()`
     - `generate_attire()`
     - `generate_entertainment()`
     - `generate_mc()`
     - `generate_traditional_ceremony()`
   - Output: SQL + JSON files

### Modified Files:

1. **`app.py`**
   - Import `WeddingPackagePlanner`
   - Inisialisasi `package_planner` di global
   - Update `process_query()` untuk handle paket request
   - Return format berbeda untuk paket vs single category

---

## 🎨 Example Output

### Package Response Format (JSON):

```json
{
  "success": true,
  "is_package": true,
  "query": "rekomendasi pernikahan adat sunda di bandung",
  "ai_result": {
    "intent": "cari_rekomendasi_paket",
    "confidence": 0.96,
    "slots": {
      "tema": "sunda",
      "lokasi": "bandung"
    }
  },
  "package": {
    "categories_found": ["Venue", "Catering", "Decoration", ...],
    "categories_missing": [],
    "total_categories": 9,
    "price_info": {
      "min_package_price": 102500000,
      "max_package_price": 112000000,
      "avg_package_price": 105666666,
      "breakdown": { ... }
    },
    "budget_status": "fully_affordable",
    "package_tiers": {
      "budget": { "items": {...}, "total_price": 102500000 },
      "standard": { "items": {...}, "total_price": 105000000 },
      "premium": { "items": {...}, "total_price": 112000000 }
    }
  },
  "categories_detail": {
    "Venue": { "items": [...], "total_items": 3 },
    "Catering": { "items": [...], "total_items": 3 },
    ...
  }
}
```

---

## 💡 Best Practices

### For Package Queries:
1. **Sebutkan tema**: "adat sunda", "adat jawa", "adat betawi"
2. **Sebutkan lokasi**: "di bandung", "di yogyakarta", "di jakarta"
3. **Sebutkan budget**: "budget 150-200 juta"
4. **Sebutkan jumlah tamu**: "untuk 500 tamu"

### Good Query Examples:

```
✅ "rekomendasi pernikahan adat sunda di bandung budget 150 juta untuk 500 tamu"
✅ "paket pernikahan lengkap adat jawa di solo budget 100-150 juta"
✅ "rancang acara pernikahan adat betawi di jakarta untuk 400 tamu"
✅ "estimasi budget pernikahan sunda di bandung"
```

### Bad Query Examples:

```
❌ "pernikahan"  (terlalu umum)
❌ "cari vendor"  (tidak spesifik)
❌ "di jakarta"  (tidak lengkap)
```

---

## 🎯 Next Steps (Future Enhancements)

1. **Web UI Update**: Tambahkan tampilan khusus untuk package view
2. **Package Comparison**: Compare multiple packages side-by-side
3. **Custom Package**: User bisa mix-and-match items dari berbagai tier
4. **Save Package**: User bisa save dan export paket pilihan
5. **Share Package**: Generate shareable link untuk paket
6. **More Themes**: Tambah tema bali, minang, bugis, dll
7. **Vendor Rating**: Tambah rating dan review vendor
8. **Real Photos**: Tambah foto untuk setiap item

---

## 📞 Support

Jika ada pertanyaan atau issue:
1. Check `KATEGORI_GUIDE.md` untuk panduan kategori
2. Check `QUICKSTART.md` untuk panduan cepat
3. Check `UPDATE_LOG.md` untuk changelog

---

**Happy Wedding Planning! 🎊💒**
