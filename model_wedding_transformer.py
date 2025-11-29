import json
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "indolem/indobert-base-uncased"  # bisa diganti model Indo lain


class SimpleWeddingIntentModel:
    """Contoh wrapper Transformer untuk mengenali intent (demo sederhana).

    Catatan: di contoh ini kita TIDAK benar-benar fine-tune ke dataset,
    hanya pakai model pre-trained sebagai encoder dan manual rule di atas output.
    Untuk production, sebaiknya di-fine-tune ke dataset_pertanyaan_wedding.csv.
    """

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Kita pakai model classification generic; untuk demo, num_labels=2 dummy
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
        self.model.eval()

    @staticmethod
    def _extract_slots_by_rule(text: str) -> Dict:
        """Ekstrak tema, lokasi, budget, jumlah_tamu, tipe_acara, venue, waktu dengan rule sederhana.

        Ini bisa diganti/ditingkatkan dengan model sequence labeling berbasis Transformer.
        """
        lower = text.lower()
        slots: Dict = {
            "tema": None,
            "lokasi": None,
            "budget_min": None,
            "budget_max": None,
            "jumlah_tamu": None,
            "tipe_acara": None,
            "venue": None,
            "waktu": None,
        }

        # Tema (simple keyword)
        if "sunda" in lower:
            slots["tema"] = "sunda"
        elif "jawa" in lower:
            slots["tema"] = "jawa"
        elif "rustic" in lower:
            slots["tema"] = "rustic"
        elif "internasional" in lower or "international" in lower:
            slots["tema"] = "internasional"
        elif "minimalis" in lower:
            slots["tema"] = "minimalis"
        elif "modern" in lower:
            slots["tema"] = "modern"

        # Lokasi (contoh beberapa kota; bisa ditambah sesuai data nyata)
        for city in [
            "bandung",
            "jakarta",
            "bogor",
            "depok",
            "bekasi",
            "bali",
            "surabaya",
            "malang",
            "garut",
            "cirebon",
            "tasikmalaya",
            "pekanbaru",
            "medan",
            "makassar",
            "yogya",
            "yogyakarta",
        ]:
            if city in lower:
                slots["lokasi"] = city
                break

        # Tipe acara
        if "lamaran" in lower and "akad" in lower:
            slots["tipe_acara"] = "lamaran+akad"
        elif "lamaran" in lower:
            slots["tipe_acara"] = "lamaran"
        elif "akad" in lower and "resepsi" in lower:
            slots["tipe_acara"] = "akad+resepsi"
        elif "akad" in lower:
            slots["tipe_acara"] = "akad"
        elif "resepsi" in lower:
            slots["tipe_acara"] = "resepsi"

        # Venue
        if "hotel" in lower:
            slots["venue"] = "hotel"
        elif "gedung" in lower:
            slots["venue"] = "gedung"
        elif "masjid" in lower:
            slots["venue"] = "masjid"
        elif "villa" in lower:
            slots["venue"] = "villa"
        elif "pantai" in lower:
            slots["venue"] = "pantai"
        elif "cafe" in lower:
            slots["venue"] = "restoran"
        elif "restoran" in lower:
            slots["venue"] = "restoran"
        elif "outdoor" in lower or "garden" in lower or "taman" in lower:
            slots["venue"] = "outdoor"

        # Waktu (sangat sederhana)
        if "malam" in lower:
            slots["waktu"] = "malam"
        elif "pagi" in lower:
            slots["waktu"] = "pagi"
        elif "minggu" in lower:
            slots["waktu"] = "minggu"
        elif "weekday" in lower:
            slots["waktu"] = "weekday"
        elif "weekend" in lower:
            slots["waktu"] = "weekend"

        # Budget (ambil angka juta paling sederhana)
        import re

        # Cari pola angka diikuti kata "juta"
        match_juta = re.findall(r"(\d+)\s*juta", lower)
        if match_juta:
            # Ambil min/max sederhana dari angka-angka juta
            amounts = [int(x) * 1_000_000 for x in match_juta]
            slots["budget_min"] = min(amounts)
            slots["budget_max"] = max(amounts)

        # Jumlah tamu (cari angka diikuti kata orang/tamu/undangan)
        match_tamu = re.search(r"(\d+)\s*(orang|tamu|undangan)", lower)
        if match_tamu:
            try:
                slots["jumlah_tamu"] = int(match_tamu.group(1))
            except ValueError:
                pass

        return slots

    @staticmethod
    def _classify_intent_by_rule(text: str) -> str:
        lower = text.lower()
        if "catering" in lower:
            return "cari_catering"
        if "dekor" in lower:
            return "cari_dekor"
        if "venue" in lower:
            return "cari_venue"
        if "vendor" in lower:
            return "cari_vendor"
        if "estimasi" in lower or "kira kira" in lower:
            return "estimasi_budget"
        if "cukup ga" in lower or "cukup nggak" in lower or "bisa?" in lower:
            return "tanya_kemungkinan"
        # default
        return "cari_rekomendasi_paket"

    def parse(self, text: str) -> Dict:
        # Di sini kita tidak pakai output label model, hanya encoding sebagai contoh;
        # Anda bisa mengembangkan jadi fine-tuning intent classifier.
        with torch.no_grad():
            _ = self.tokenizer(text, return_tensors="pt", truncation=True)
            # _output = self.model(**encoded)  # tidak dipakai di demo

        intent = self._classify_intent_by_rule(text)
        slots = self._extract_slots_by_rule(text)

        return {
            "text": text,
            "intent": intent,
            "slots": slots,
        }


def interactive_cli() -> None:
    model = SimpleWeddingIntentModel()
    print("Ketik pertanyaan (ketik 'exit' untuk keluar):")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"exit", "quit"}:
            break
        result = model.parse(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    interactive_cli()
