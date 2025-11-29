import csv
import random
import os
from typing import List, Dict, Tuple

TARGET_ROWS = 10000
OUTPUT_FILE = "dataset_pertanyaan_wedding.csv"
SEED = 123
random.seed(SEED)

# Lebih banyak adat/lokal etnis Indonesia
ADATS: List[str] = [
    "sunda","jawa","minang","batak","bugis","melayu","aceh","bali","betawi","madura",
    "dayak","toraja","banjar","sasak","ambon","maluku","papua","bima","banyumasan",
    "mandailing","bengkulu","lampung","bima-dompuh","tidung","bajau","bima","ternate",
    "tidore","alor","ntt","ntb","banjar-melayu","makassar","bugis-makassar","karo","nias",
    "bali-modern","jawa-syariah","minang-modern","betawi-modern","jawa-kraton","madura-islami"
]

INTENTS = [
    "cari_rekomendasi_paket","estimasi_budget","cari_venue","tanya_kemungkinan",
    "cari_dekor","cari_vendor","cari_catering"
]

LOKASIS = [
    # Jabodetabek detail
    "jakarta pusat","jakarta barat","jakarta timur","jakarta selatan","jakarta utara",
    "bogor","depok","bekasi","tangerang","tangerang selatan",
    # Jawa Barat
    "bandung","cimahi","sumedang","garut","cirebon","tasikmalaya","kuningan","majalengka",
    # Jawa Tengah/DIY/Jawa Timur
    "yogyakarta","sleman","bantul","kulon progo","gunungkidul","semarang","solo","surakarta",
    "magelang","salatiga","tegal","pekalongan","surabaya","sidoarjo","malang","kediri","madiun",
    # Sumatra
    "medan","binjai","padang","bukittinggi","pekanbaru","palembang","bengkulu","lampung",
    # Kalimantan
    "pontianak","banjarmasin","balikpapan","samarinda","tarakan",
    # Sulawesi
    "makassar","kendari","manado","gorontalo",
    # Bali & Nusa
    "bali","denpasar","ubud","canggu","nusa dua","lombok","mataram",
    # NTT/NTB
    "labuan bajo","kupang",
    # Maluku/Papua
    "ambon","ternate","jayapura",
    # Destinasi alam
    "puncak","sentul","anyer","carita","ciwidey","lembang"
]

TIPE_ACARA = ["akad","resepsi","akad+resepsi","lamaran","lamaran+akad","siraman","midodareni"]

VENUES = [
    "gedung","hotel","outdoor","rumah","restoran","resort","masjid","villa","pantai",
    "rooftop","cafe","ballroom","kebun","taman","aula","balai desa","pendopo","gereja"
]

WAKTUS = ["", "pagi", "siang", "sore", "malam", "minggu siang", "weekday", "sabtu malam", "minggu pagi"]

RELIGIOUS_CONTEXT = ["syar'i","muslim friendly","halal","adat + syariah","non-alkohol"]
SERVICES = [
    "catering","dekor","dokumentasi","fotografi","videografi","rias","makeup","busana","mc","band","hiburan",
    "organizer","wedding planner","undangan","souvenir","photobooth","lighting","sound system"
]

ASK_PREFIX = [
    "rekomendasi","butuh","minta","tolong carikan","saya cari","pengen","mau tanya",
    "ada paket","cari paket","butuh info","ingin tahu","bisa bantu","butuh rekomendasi"
]
PACKAGE_WORDS = [
    "paket","paketan","bundle","all in","layanan","service","paket lengkap","paket hemat","paket premium"
]
BUDGET_PHRASES = [
    "budget","sekitar","kisaran","range","kira kira","maksimal","sampai","hanya","cuma","di bawah","di atas"
]
TAMU_PHRASES = [
    "untuk {tamu} orang","buat {tamu} tamu","{tamu} undangan","sekitar {tamu} tamu","kapasitas {tamu}","{tamu} pax"
]
ADAT_CONNECTORS = ["tema","adat","konsep","style","nuansa","theme","pakem"]
QUESTION_ENDINGS = ["berapa ya?","bisa?","muat?","cukup ga?","gimana?","ok?","mungkin?","tersedia?","ada?","rekomendasi?","gimana kira kiranya?",""]

BUDGET_BRACKETS: List[Tuple[int,int]] = [
    (0, 5000000),(0, 8000000),(0,10000000),(5000000,15000000),(0,20000000),(10000000,30000000),
    (20000000,40000000),(30000000,50000000),(40000000,60000000),(50000000,70000000),
    (60000000,80000000),(70000000,90000000),(80000000,100000000),(90000000,120000000),
    (100000000,150000000),(120000000,180000000),(150000000,200000000),(180000000,250000000),
    (200000000,300000000),(250000000,400000000)
]
GUEST_COUNTS = [20,30,50,60,75,80,90,100,120,150,180,200,220,250,300,320,350,400,450,500,550,600,700]

INTENT_IGNORE_BUDGET = {"estimasi_budget"}
INTENT_POSSIBILITY = {"tanya_kemungkinan"}

# Variasi kalimat untuk penambahan layanan
SERVICE_EXPRESSIONS = [
    "termasuk {service}","dengan {service}","+ {service}","sekalian {service}","sama {service}",
    "include {service}","pakai {service}","butuh {service}","minta {service}"
]

# Variasi konteks tambahan
SEASON_CONTEXT = ["musim hujan","musim kemarau","cuaca tidak menentu","backup indoor"]
PAYMENT_CONTEXT = ["dp bertahap","cicilan","bayar lunas","termin 3x","cash"]
COUPLE_CONTEXT = ["nikah campur adat","campuran adat {a1}-{a2}","adat {a1} tapi sentuhan {a2}"]


def load_existing(path: str) -> List[Dict]:
    rows = []
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def synthesize_text(intent: str, tema: str, lokasi: str, bmin: int, bmax: int, tamu: int, tipe: str, venue: str, waktu: str) -> str:
    prefix = random.choice(ASK_PREFIX)
    adat_conn = random.choice(ADAT_CONNECTORS)
    pkg_word = random.choice(PACKAGE_WORDS)
    tamu_phrase = random.choice(TAMU_PHRASES).format(tamu=tamu)
    ending = random.choice(QUESTION_ENDINGS)

    # Budget phrase
    budget_phrase = ""
    if intent in INTENT_IGNORE_BUDGET:
        budget_phrase = ""
    else:
        if bmin == 0 and bmax > 0:
            style = random.choice(["maksimal {}", "limit {}", "plafon {}", "sampai {}", "upper {}", "<= {}"])
            budget_phrase = f"{random.choice(BUDGET_PHRASES)} {style.format(bmax):s}".strip()
        elif bmin > 0 and bmax > 0:
            if random.random() < 0.5:
                budget_phrase = f"{random.choice(BUDGET_PHRASES)} {bmin//1000000}-{bmax//1000000} jutaan"
            else:
                budget_phrase = f"{random.choice(BUDGET_PHRASES)} {bmax//1000000} juta"
        elif bmin == 0 and bmax == 0:
            budget_phrase = random.choice([
                "butuh estimasi budget","kira kira biayanya","estimasi biayanya","range harganya"
            ])

    if budget_phrase and random.random() < 0.35 and bmax > 0:
        budget_phrase += f" (~{bmax//1000000}M)"

    waktu_part = f" {waktu}" if waktu else ""

    # Service additions randomly
    extra = ""
    if random.random() < 0.6:
        service = random.choice(SERVICES)
        extra = " " + random.choice(SERVICE_EXPRESSIONS).format(service=service)

    # Religious context occasionally
    relig = ""
    if random.random() < 0.3:
        relig = " " + random.choice(RELIGIOUS_CONTEXT)

    # Season/payment context sometimes
    season = ""
    if random.random() < 0.2:
        season = " " + random.choice(SEASON_CONTEXT)
    payment = ""
    if random.random() < 0.15:
        payment = " " + random.choice(PAYMENT_CONTEXT)

    # Couple mixed adat context
    couple = ""
    if random.random() < 0.2:
        a1, a2 = random.sample(ADATS, 2)
        couple = " " + random.choice(COUPLE_CONTEXT).format(a1=a1, a2=a2)

    if intent == "cari_venue":
        core = f"{prefix} venue {venue} di {lokasi} {tamu_phrase} {budget_phrase}".strip()
    elif intent == "cari_vendor":
        core = f"{prefix} vendor pernikahan {tema} di {lokasi} {tamu_phrase} {budget_phrase}".strip()
    elif intent == "cari_catering":
        core = f"{prefix} catering {tema} di {lokasi} {tamu_phrase} {budget_phrase}".strip()
    elif intent == "cari_dekor":
        core = f"{prefix} dekor {tema} {tipe} di {lokasi} {tamu_phrase} {budget_phrase}".strip()
    elif intent == "estimasi_budget":
        core = f"{prefix} {pkg_word} {tipe} {adat_conn} {tema} di {lokasi} {tamu_phrase} untuk estimasi biaya".strip()
    elif intent == "tanya_kemungkinan":
        core = f"{prefix} {pkg_word} {tipe} {adat_conn} {tema} di {lokasi} {tamu_phrase} {budget_phrase} memungkinkan?".strip()
    else:  # cari_rekomendasi_paket
        core = f"{prefix} {pkg_word} {tipe} {adat_conn} {tema} di {lokasi} {tamu_phrase} {budget_phrase}".strip()

    if venue and intent != "cari_venue" and random.random() < 0.6:
        core += f" venue {venue}"

    core += relig + extra + season + payment + couple

    if waktu_part:
        core += waktu_part

    core = ' '.join(core.split())
    if ending and not core.endswith('?') and random.random() < 0.5:
        core += f" {ending}".strip()

    return core


def generate_row(intent: str, tema: str, lokasi: str) -> Dict:
    tamu = random.choice(GUEST_COUNTS)
    tipe = random.choice(TIPE_ACARA)
    venue = random.choice(VENUES)
    waktu = random.choice(WAKTUS)

    if intent in INTENT_IGNORE_BUDGET:
        bmin, bmax = 0, 0
    else:
        bmin, bmax = random.choice(BUDGET_BRACKETS)
        if bmin > 0 and random.random() < 0.3:
            span = bmax - bmin
            if span > 5_000_000:
                mid = bmin + span // 2
                bmax = mid + span // 4
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
    data = existing[:]
    print(f"Existing rows: {len(existing)}")

    while len(data) < TARGET_ROWS:
        intent = random.choice(INTENTS)
        tema = random.choice(ADATS)
        lokasi = random.choice(LOKASIS)
        data.append(generate_row(intent, tema, lokasi))

    # Deduplicate by text
    seen = set()
    deduped = []
    for r in data:
        t = r['text']
        if t not in seen:
            deduped.append(r)
            seen.add(t)

    regen_attempts = 0
    while len(deduped) < TARGET_ROWS and regen_attempts < 15000:
        regen_attempts += 1
        row = generate_row(random.choice(INTENTS), random.choice(ADATS), random.choice(LOKASIS))
        if row['text'] not in seen:
            deduped.append(row)
            seen.add(row['text'])

    print(f"Final row count: {len(deduped)}")

    fieldnames = ['text','intent','tema','lokasi','budget_min','budget_max','jumlah_tamu','tipe_acara','venue','waktu']
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Wrote {len(deduped)} rows to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
