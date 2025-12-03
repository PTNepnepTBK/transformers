# Kategori Database & Contoh Query

## 📋 Kategori yang Tersedia

Database memiliki 9 kategori utama untuk item pernikahan:

| ID | Kategori | Deskripsi | Keyword |
|----|----------|-----------|---------|
| `3c476203-...` | **Venue** | Tempat pernikahan dari berbagai daerah | venue, tempat, gedung, ballroom, hotel, villa, pantai, outdoor |
| `33126b96-...` | **Catering** | Makanan dan minuman untuk acara | catering, makanan, menu, prasmanan, nasi kotak, gubukan |
| `b01d482d-...` | **Decoration** | Dekorasi pernikahan dari Indonesia | dekorasi, dekor, pelaminan, hiasan, bunga |
| `1b2f2a98-...` | **MUA** | Make Up Artist dan hair do | mua, makeup, make up, riasan, rias, hair do |
| `b4c13e15-...` | **Docummentation** | Dokumentasi fotografi dan videografi | foto, video, dokumentasi, fotografi, videografi, photographer, cinematic |
| `ae51cb01-...` | **Attire** | Busana khas ciri Indonesia | baju, busana, kebaya, gaun, attire, pakaian, sewa baju, jas |
| `bef9f40f-...` | **Entertainment** | Hiburan seperti musik dan keyboard | band, musik, entertainment, hiburan, penyanyi, dj, gamelan, angklung |
| `b123275f-...` | **Master of Ceremony** | Pembawa Acara | mc, pembawa acara, master of ceremony, host |
| `c5b14377-...` | **Traditional Ceremony** | Upacara tradisional | upacara, siraman, midodareni, ceremony, adat, ritual, tradisi, ngeuyeuk seureuh |

## 🎯 Intent AI dan Mapping Kategori

| Intent | Mapping Kategori | Deskripsi |
|--------|------------------|-----------|
| `cari_rekomendasi_paket` | **All** | Mencari paket lengkap, semua kategori ditampilkan |
| `estimasi_budget` | **All** | Estimasi budget, semua kategori ditampilkan |
| `cari_venue` | **Venue** | Khusus mencari tempat pernikahan |
| `cari_dekor` | **Decoration** | Khusus mencari dekorasi |
| `cari_catering` | **Catering** | Khusus mencari catering |
| `cari_vendor` | **Auto-detect** | Otomatis deteksi dari keyword |
| `tanya_kemungkinan` | **All** | Pertanyaan umum, semua kategori |

## 📝 Contoh Query per Kategori

### 🏛️ Venue
```
✅ cari venue pernikahan di bandung budget 50 juta
✅ rekomendasi tempat pernikahan outdoor di jakarta
✅ gedung pernikahan adat sunda di bandung untuk 500 tamu
✅ ballroom hotel di jakarta pusat
✅ villa pernikahan di bali untuk intimate wedding
```

### 🍽️ Catering
```
✅ paket catering sunda untuk 500 orang di bandung
✅ catering pernikahan menu jawa untuk 300 tamu
✅ catering nusantara budget 40 juta
✅ gubukan nasi kebuli untuk acara pernikahan
✅ prasmanan seafood untuk 600 tamu
```

### 🎨 Decoration
```
✅ dekorasi pernikahan adat sunda di bandung
✅ dekorasi minimalis modern untuk resepsi
✅ pelaminan adat jawa budget 25 juta
✅ dekorasi rustic outdoor garden
✅ dekorasi ballroom mewah
```

### 💄 MUA (Make Up Artist)
```
✅ MUA pengantin adat sunda di bandung
✅ makeup artist untuk pernikahan jawa
✅ riasan pengantin sunda siger
✅ paket MUA akad dan resepsi
✅ make up bridesmaid untuk 5 orang
```

### 📸 Docummentation
```
✅ dokumentasi foto video pernikahan sunda di bandung
✅ paket foto pernikahan premium
✅ video cinematic untuk pernikahan
✅ photographer dan videographer untuk wedding
✅ dokumentasi pre-wedding outdoor
```

### 👗 Attire (Busana)
```
✅ sewa kebaya sunda untuk pengantin di bandung
✅ busana pengantin adat jawa
✅ gaun pernikahan modern
✅ sewa baju adat sunda
✅ kebaya akad nikah
```

### 🎵 Entertainment
```
✅ band akustik sunda untuk pernikahan di bandung
✅ musik gamelan jawa untuk acara adat
✅ dj untuk resepsi malam
✅ hiburan pernikahan dengan angklung
✅ penyanyi untuk acara pernikahan
```

### 🎤 Master of Ceremony
```
✅ MC pernikahan adat sunda di bandung
✅ pembawa acara untuk wedding
✅ MC bilingual indonesia-english
✅ host untuk acara pernikahan
✅ MC couple untuk akad dan resepsi
```

### 🕌 Traditional Ceremony
```
✅ upacara adat sunda lengkap di bandung
✅ siraman dan midodareni adat jawa
✅ upacara ngeuyeuk seureuh sunda
✅ paket upacara adat lengkap
✅ ritual pernikahan tradisional
```

## 🔍 Auto-Detection dari Keyword

Sistem akan otomatis mendeteksi kategori jika keyword tertentu ditemukan di query:

```python
Keyword → Kategori Detected:
"mua", "makeup", "riasan" → MUA
"foto", "video", "dokumentasi" → Docummentation
"kebaya", "gaun", "busana" → Attire
"mc", "pembawa acara" → Master of Ceremony
"band", "musik", "hiburan" → Entertainment
"upacara", "siraman", "adat" → Traditional Ceremony
"venue", "tempat", "gedung" → Venue
"catering", "makanan" → Catering
"dekorasi", "dekor" → Decoration
```

## 🎨 Query Multi-Kategori

Untuk query yang membutuhkan **paket lengkap**, sistem akan mencari dari **semua kategori**:

```
✅ rekomendasi pernikahan adat sunda di bandung budget 200-300 juta untuk 500 tamu
   → Akan mencari: Venue + Catering + Decoration + MUA + dll

✅ paket pernikahan lengkap adat jawa budget 150 juta
   → Akan mencari semua kategori yang relevan

✅ estimasi budget pernikahan untuk 400 tamu di jakarta
   → Akan menampilkan semua kategori untuk estimasi
```

## 🧠 Cara Kerja AI

1. **Input Query**: User mengetik pertanyaan
2. **Intent Classification**: AI mengklasifikasi intent (7 jenis)
3. **Slot Extraction**: Ekstrak tema, lokasi, budget, jumlah tamu
4. **Category Detection**: 
   - Jika intent jelas → langsung map ke kategori
   - Jika intent umum → deteksi dari keyword
5. **Database Query**: Filter items berdasarkan kategori + slots
6. **Clustering**: K-Means untuk ranking recommendations
7. **Output**: Top 10 rekomendasi paling relevan

## 📊 Contoh Proses Lengkap

### Input:
```
"MUA pengantin adat sunda di bandung budget 4-6 juta"
```

### Proses:
1. **Intent**: `cari_vendor` (confidence: 96%)
2. **Slots Extracted**:
   - tema: sunda
   - lokasi: bandung
   - budget_min: 4000000
   - budget_max: 6000000
3. **Category Detection**: "MUA" (dari keyword "MUA")
4. **Database Query**: 
   ```sql
   SELECT * FROM items 
   WHERE category = 'MUA'
   AND (name LIKE '%sunda%' OR description LIKE '%sunda%')
   AND (vendor LIKE '%bandung%' OR description LIKE '%bandung%')
   AND price >= 3200000 AND price <= 7200000  -- toleransi ±20%
   ```
5. **Clustering**: K-Means dengan 3 clusters
6. **Result**: 5-10 MUA yang paling cocok

### Output:
```
✅ Paket MUA Pengantin Adat Sunda - Rp 4,500,000
✅ Paket MUA Sunda Siger Premium - Rp 6,000,000
✅ MUA Sunda Modern Bandung - Rp 4,000,000
✅ Paket MUA Akad dan Resepsi Sunda - Rp 8,000,000 (sedikit over)
...
```

## 💡 Tips untuk Query yang Baik

1. **Sebutkan kategori** jika spesifik: "MUA", "venue", "catering"
2. **Tambahkan tema**: "adat sunda", "minimalis", "modern"
3. **Sebutkan lokasi**: "bandung", "jakarta", "bali"
4. **Tambahkan budget**: "50 juta", "100-150 juta"
5. **Sebutkan jumlah tamu**: "500 tamu", "300 orang"

### Query Kurang Baik ❌
```
pernikahan di jakarta
cari vendor
rekomendasi
```

### Query Baik ✅
```
cari venue pernikahan di jakarta budget 80 juta untuk 600 tamu
MUA pengantin adat sunda di bandung
paket catering menu jawa untuk 400 orang budget 35 juta
dekorasi minimalis modern di jakarta budget 20-30 juta
```

---

**Note**: Sistem memiliki **flexible matching** dengan toleransi ±20% untuk budget dan akan otomatis relax filter jika tidak ada hasil yang ditemukan!
