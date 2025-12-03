# Quick Start - Wedding Package System

## 🚀 Setup Database (One-time)

```bash
# 1. Generate 345 wedding items data
python expand_database.py

# 2. Import ke database
python import_data.py generated_data_all.sql
```

## 🎯 Usage

### Option 1: Web Interface (Recommended)

```bash
python app.py
# Buka: http://localhost:5000
```

**Query untuk Paket Pernikahan:**
```
rekomendasi pernikahan adat sunda di bandung
paket pernikahan adat jawa di yogyakarta budget 150 juta
acara pernikahan adat betawi di jakarta untuk 500 tamu
```

### Option 2: CLI

```bash
# Test package planner
python package_planner.py

# Or use main.py
python main.py
```

## 📋 Data Generated

### 3 Tema x 9 Kategori = 345 Items

**Tema:**
- 🎨 Sunda (115 items) - Bandung, Bogor, dll
- 🎭 Jawa (115 items) - Yogyakarta, Solo, dll  
- 🏛️ Betawi (115 items) - Jakarta, Tangerang

**Kategori (9):**
1. Venue (tempat)
2. Catering (makanan)
3. Decoration (dekorasi)
4. MUA (makeup artist)
5. Documentation (foto/video)
6. Attire (busana)
7. Entertainment (hiburan)
8. Master of Ceremony (MC)
9. Traditional Ceremony (upacara adat)

## 💚💙💛 Package Tiers

Sistem otomatis membuat **3 paket**:
- 💚 **Budget**: Pilihan termurah (~100 juta)
- 💙 **Standard**: Pilihan tengah (~105 juta)
- 💛 **Premium**: Pilihan terbaik (~112 juta)

## 📝 Query Examples

### ✅ Package Queries (Multi-kategori):
```
"rekomendasi pernikahan adat sunda di bandung"
"paket pernikahan lengkap adat jawa budget 150 juta"
"rancang pernikahan adat betawi di jakarta untuk 500 tamu"
"estimasi budget pernikahan sunda di bandung"
```

### ❌ Single Category Queries:
```
"cari venue pernikahan di bandung"
"MUA pengantin sunda di bandung"
"dokumentasi foto video pernikahan"
```

## 🎯 Features

✅ Auto-detect paket request  
✅ 9 kategori vendor  
✅ 3 package tiers (Budget/Standard/Premium)  
✅ Price breakdown per kategori  
✅ Budget compatibility check  
✅ Smart category detection  
✅ 680+ items database  
✅ 3 tema budaya (Sunda/Jawa/Betawi)  

## 📊 Database Stats

```
Total: 680+ items
Price range: Rp 18,000 - Rp 180,000,000
Average: Rp 23,677,467

By Theme:
- Sunda: 115 items
- Jawa: 115 items  
- Betawi: 115 items
- Others: 335 items
```

---

**Ready to plan your dream wedding! 🎊**
