# Testing Guide - Wedding Recommendation System

## 🧪 Query Test Cases

### 1. Package Queries (Multi-Category)

#### Test 1A: Basic Package - Sunda
```
Query: "rekomendasi pernikahan adat sunda di bandung"

Expected:
✅ Detected as package request
✅ 9 categories searched
✅ Display 3 package tiers
✅ Show items grouped by category
```

#### Test 1B: Basic Package - Jawa
```
Query: "paket pernikahan adat jawa di yogyakarta"

Expected:
✅ Detected as package request
✅ Categories: Venue, Catering, Decoration, MUA, Documentation, Attire, Entertainment, MC, Traditional Ceremony
✅ Price breakdown available
```

#### Test 1C: Package with Budget
```
Query: "rekomendasi pernikahan adat sunda di bandung budget 150 juta"

Expected:
✅ Budget status: "fully_affordable" or "over_budget"
✅ User budget displayed
✅ 3 tiers with prices
```

#### Test 1D: Package with Budget & Guests
```
Query: "acara pernikahan adat betawi di jakarta budget 200 juta untuk 500 tamu"

Expected:
✅ All slots extracted (tema, lokasi, budget, jumlah_tamu)
✅ Budget comparison shown
✅ Complete package displayed
```

---

### 2. Single Category Queries

#### Test 2A: Venue Search
```
Query: "cari venue pernikahan di bandung budget 50 juta"

Expected:
✅ NOT detected as package
✅ Category: Venue only
✅ K-Means clustering shown
✅ Cluster visualization graph
✅ Top 10 recommendations
```

#### Test 2B: MUA Search
```
Query: "MUA pengantin adat sunda di bandung"

Expected:
✅ Category: MUA
✅ Tema: sunda, Lokasi: bandung detected
✅ Clustering analysis
✅ Recommendations with relevance score
```

#### Test 2C: Documentation Search
```
Query: "dokumentasi foto video pernikahan sunda di bandung"

Expected:
✅ Category: Documentation
✅ Filter by tema & lokasi
✅ Clustering display
```

#### Test 2D: Catering Search
```
Query: "paket catering sunda untuk 500 orang di bandung"

Expected:
✅ Category: Catering
✅ Jumlah tamu: 500 extracted
✅ Price per pax consideration
```

---

## 📊 Expected UI Elements

### For Package Requests:
```
Step 1: AI Intent Classification
   ✅ Intent: cari_rekomendasi_paket
   ✅ Confidence: ~96%
   ✅ Slots: tema, lokasi, budget (if any)

Step 2: Database Query
   ✅ "9 categories found"
   ✅ "Package Request (9/9)"

Step 3: Package Price Information
   ✅ Min Package: Rp xxx
   ✅ Max Package: Rp xxx
   ✅ Budget Status: [icon] status
   
   Cards:
   💚 Budget Package
   💙 Standard Package
   💛 Premium Package

Step 4: Items by Category
   Venue (3 options)
   Catering (3 options)
   Decoration (3 options)
   ...
```

### For Single Category:
```
Step 1: AI Intent Classification
   ✅ Intent: cari_venue / cari_vendor / etc
   ✅ Confidence: ~XX%
   ✅ Slots: extracted values

Step 2: Database Query
   ✅ XX items found
   ✅ Category: [CategoryName]

Step 3: K-Means Clustering
   ✅ Cluster cards with size & avg price
   ✅ Best cluster marked with ⭐
   ✅ Visualization graph (if available)

Step 4: Top Recommendations
   ✅ Cards with name, vendor, price, score
   ✅ Category badge
   ✅ Cluster badge
```

---

## 🔍 Validation Checklist

### Backend (Console Log):
- [ ] `🎊 Detected package request` (for package)
- [ ] `🎨 Merancang paket pernikahan dengan X kategori...`
- [ ] `✓ Ditemukan X pilihan` for each category
- [ ] `HTTP 200` response
- [ ] No Python errors

### Frontend (Browser):
- [ ] No JavaScript console errors
- [ ] All 4 steps displayed
- [ ] Cards rendered correctly
- [ ] Images loaded (if any)
- [ ] Prices formatted with Rp and thousand separator
- [ ] Smooth scroll to results

### Data Validation:
- [ ] Intent classification correct
- [ ] Slots extracted correctly
- [ ] Budget comparison accurate
- [ ] Categories returned match query
- [ ] Prices within reasonable range
- [ ] Relevance scores make sense

---

## 🐛 Troubleshooting

### Issue: JavaScript Error
```
Error: Cannot read properties of undefined
```
**Solution:** Check response format, ensure `is_package` flag set correctly

### Issue: No Results Found
```
0 items found
```
**Solution:** 
1. Check database has data for that tema/lokasi
2. Run: `python import_data.py generated_data_all.sql`
3. Verify: 680+ items in database

### Issue: Wrong Category Detection
```
Query: "MUA pengantin" → Category: Venue (wrong!)
```
**Solution:** Check keyword detection in `app.py` line ~160

### Issue: Budget Status Wrong
```
Budget 200M but status "over_budget" for 100M package
```
**Solution:** Check `package_planner.py` budget comparison logic

---

## 📝 Manual Test Script

```bash
# 1. Start server
python app.py

# 2. Open browser
http://localhost:5000

# 3. Test queries (copy paste):

# Package queries:
rekomendasi pernikahan adat sunda di bandung
paket pernikahan adat jawa di yogyakarta budget 150 juta
acara pernikahan adat betawi di jakarta untuk 500 tamu

# Single category queries:
cari venue pernikahan di bandung budget 50 juta
MUA pengantin sunda di bandung
dokumentasi foto video pernikahan
paket catering sunda untuk 500 orang

# 4. Verify each query:
✅ No console errors
✅ Results displayed correctly
✅ Prices formatted
✅ All data shown
```

---

## ✅ Success Criteria

**All tests pass when:**
1. ✅ No JavaScript errors in browser console
2. ✅ No Python errors in terminal
3. ✅ Package queries show category breakdown
4. ✅ Single queries show clustering
5. ✅ Budget comparison works correctly
6. ✅ All prices formatted with Rp
7. ✅ Relevance scores displayed
8. ✅ UI responsive and smooth
9. ✅ Data accurate and relevant
10. ✅ HTTP 200 for all requests

---

**Current Status: ✅ ALL SYSTEMS OPERATIONAL**

Server: `http://localhost:5000`  
Database: 680+ items ready  
Package System: Fully functional  
