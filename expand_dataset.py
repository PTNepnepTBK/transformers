import csv
import random
import os

# Target minimum rows
TARGET_ROWS = 5000
OUTPUT_FILE = "dataset_pertanyaan_wedding.csv"  # overwrite same file
SEED = 42
random.seed(SEED)

# Controlled vocabularies (extracted + expanded)
INTENTS = [
    "cari_rekomendasi_paket","estimasi_budget","cari_venue","tanya_kemungkinan",
    "cari_dekor","cari_vendor","cari_catering"
]
TEMAS = [
    "sunda","jawa","rustic","modern","minimalis","internasional","mix_sunda_jawa",
    "bugis","minang","melayu","batak"
]
LOKASIS = [
    "bandung","jakarta barat","jakarta timur","jakarta selatan","jakarta pusat","bogor",
    "depok","bekasi","cirebon","garut","tasikmalaya","bali","surabaya","malang",
    "semarang","yogyakarta","makassar","padang","pekanbaru","medan","surakarta","sentul","puncak"
]
TIPE_ACARA = ["akad","resepsi","akad+resepsi","lamaran","lamaran+akad"]
VENUES = [
    "gedung","hotel","outdoor","rumah","restoran","resort","masjid","villa","pantai",
    "rooftop","cafe","ballroom","kebun","taman"
]
WAKTUS = ["", "pagi", "siang", "malam", "minggu siang", "weekday", "sabtu malam"]

# Phrase pools for natural language synthesis
ASK_PREFIX = [
    "rekomendasi","butuh","minta","tolong carikan","saya cari","pengen","mau tanya",
    "ada paket","cari paket","butuh info","ingin tahu","bisa bantu","butuh rekomendasi"
]
PACKAGE_WORDS = [
    "paket","paketan","bundle","all in","layanan","service","paket lengkap"
]
BUDGET_PHRASES = [
    "budget","sekitar","kisaran","range","kira kira","maksimal","sampai","hanya","cuma"
]
TAMU_PHRASES = [
    "untuk {tamu} orang","buat {tamu} tamu","{tamu} undangan","sekitar {tamu} tamu","kapasitas {tamu}","{tamu} pax"
]
ADAT_CONNECTORS = ["tema","adat","konsep","style","nuansa","theme"]
QUESTION_ENDINGS = ["berapa ya?","bisa?","muat?","cukup ga?","gimana?","ok?","mungkin?","tersedia?","ada?","rekomendasi?","gimana kira kiranya?",""]

# Some budget ranges typical for Indonesian wedding scenarios
BUDGET_BRACKETS = [
    (0, 5000000),(0, 8000000),(0,10000000),(5000000,15000000),(0,20000000),(10000000,30000000),
    (20000000,40000000),(30000000,50000000),(40000000,60000000),(50000000,70000000),
    (60000000,80000000),(70000000,90000000),(80000000,100000000),(90000000,120000000),
    (100000000,150000000),(120000000,180000000),(150000000,200000000),(180000000,250000000),
    (200000000,300000000)
]
# Guest count typical distribution
GUEST_COUNTS = [20,30,50,80,100,120,150,180,200,250,300,350,400,450,500,550,600]

# For intents that ignore budget max or both (estimasi_budget) we handle specially
INTENT_IGNORE_BUDGET = {"estimasi_budget"}
INTENT_POSSIBILITY = {"tanya_kemungkinan"}


def load_existing(path):
    rows = []
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def synthesize_text(intent, tema, lokasi, bmin, bmax, tamu, tipe, venue, waktu):
    prefix = random.choice(ASK_PREFIX)
    adat_conn = random.choice(ADAT_CONNECTORS)
    pkg_word = random.choice(PACKAGE_WORDS)
    tamu_phrase = random.choice(TAMU_PHRASES).format(tamu=tamu)
    ending = random.choice(QUESTION_ENDINGS)

    # Compose budget phrase
    budget_phrase = ""
    if intent in INTENT_IGNORE_BUDGET:
        budget_phrase = ""
    else:
        if bmin == 0 and bmax > 0:
            style = random.choice(["maksimal {}", "limit {}", "plafon {}", "sampai {}", "upper {}"])
            budget_phrase = f"{random.choice(BUDGET_PHRASES)} {style.format(bmax):s}".strip()
        elif bmin > 0 and bmax > 0:
            budget_phrase = f"{random.choice(BUDGET_PHRASES)} {bmin//1000000}-{bmax//1000000} jutaan"
        elif bmin == 0 and bmax == 0:
            # unknown budget -> ask estimation
            budget_phrase = random.choice([
                "butuh estimasi budget","kira kira biayanya","estimasi biayanya","range harganya"
            ])

    # Insert currency clarity sometimes
    if budget_phrase and random.random() < 0.4 and bmax > 0:
        budget_phrase += f" (~{bmax//1000000}M)"

    waktu_part = f" {waktu}" if waktu else ""

    # Build core text variations based on intent
    if intent == "cari_venue":
        core = f"{prefix} venue {venue} di {lokasi} {tamu_phrase} {budget_phrase}".strip()
    elif intent == "cari_vendor":
        core = f"{prefix} vendor pernikahan {tema} di {lokasi} {tamu_phrase} {budget_phrase}".strip()
    elif intent == "cari_catering":
        core = f"{prefix} catering {tema} di {lokasi} {tamu_phrase} {budget_phrase}".strip()
    elif intent == "cari_dekor":
        core = f"{prefix} dekor {tema} {tipe} di {lokasi} {tamu_phrase} {budget_phrase}".strip()
    elif intent == "estimasi_budget":
        core = f"{prefix} {pkg_word} {tipe} {tema} di {lokasi} {tamu_phrase} untuk estimasi biaya".strip()
    elif intent == "tanya_kemungkinan":
        core = f"{prefix} {pkg_word} {tipe} {tema} di {lokasi} {tamu_phrase} {budget_phrase} memungkinkan?".strip()
    else:  # cari_rekomendasi_paket
        core = f"{prefix} {pkg_word} {tipe} {adat_conn} {tema} di {lokasi} {tamu_phrase} {budget_phrase}".strip()

    if venue and intent != "cari_venue" and random.random() < 0.6:
        core += f" venue {venue}"

    if waktu_part:
        core += waktu_part

    core = ' '.join(core.split())  # normalize spaces
    if ending and not core.endswith('?') and random.random() < 0.5:
        if ending:
            core += f" {ending}".strip()
    return core


def generate_row(intent, tema, lokasi):
    tamu = random.choice(GUEST_COUNTS)
    tipe = random.choice(TIPE_ACARA)
    venue = random.choice(VENUES)
    waktu = random.choice(WAKTUS)

    if intent in INTENT_IGNORE_BUDGET:
        bmin, bmax = 0, 0
    else:
        bmin, bmax = random.choice(BUDGET_BRACKETS)
        # Occasionally create explicit min-max narrowing
        if bmin > 0 and random.random() < 0.3:
            # narrow range
            span = bmax - bmin
            if span > 5_000_000:
                mid = bmin + span // 2
                bmax = mid + span // 4
    # Ensure logical ordering
    if bmax < bmin:
        bmin, bmax = bmax, bmin

    text = synthesize_text(intent, tema, lokasi, bmin, bmax, tamu, tipe, venue, waktu)

    return {
        'text': text,
        'intent': intent,
        'tema': tema,
        'lokasi': lokasi,
        'budget_min': bmin,
        'budget_max': bmax,
        'jumlah_tamu': tamu,
        'tipe_acara': tipe,
        'venue': venue,
        'waktu': waktu
    }


def main():
    existing = load_existing(OUTPUT_FILE)
    current_count = len(existing)
    print(f"Existing rows: {current_count}")

    # Keep existing rows, then generate new ones until reaching target
    data = existing[:]

    # To ensure coverage, iterate combinations with randomness
    while len(data) < TARGET_ROWS:
        intent = random.choice(INTENTS)
        tema = random.choice(TEMAS)
        lokasi = random.choice(LOKASIS)
        row = generate_row(intent, tema, lokasi)
        data.append(row)

    # Deduplicate by text (optional) while preserving first occurrences
    seen = set()
    deduped = []
    for r in data:
        t = r['text']
        if t not in seen:
            deduped.append(r)
            seen.add(t)
    # If dedup removed too many, regenerate until count restored
    regen_attempts = 0
    while len(deduped) < TARGET_ROWS and regen_attempts < 5000:
        regen_attempts += 1
        row = generate_row(random.choice(INTENTS), random.choice(TEMAS), random.choice(LOKASIS))
        if row['text'] not in seen:
            deduped.append(row)
            seen.add(row['text'])

    print(f"Final row count: {len(deduped)}")

    # Write CSV
    fieldnames = ['text','intent','tema','lokasi','budget_min','budget_max','jumlah_tamu','tipe_acara','venue','waktu']
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Wrote {len(deduped)} rows to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
