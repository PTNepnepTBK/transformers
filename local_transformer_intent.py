import os
import json
import math
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv

# Simple whitespace tokenizer + vocabulary builder
class SimpleTokenizer:
    def __init__(self, vocab: Dict[str, int], unk_token: str = "<unk>", pad_token: str = "<pad>"):
        self.vocab = vocab
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.unk_id = self.vocab.get(self.unk_token, 0)
        self.pad_id = self.vocab.get(self.pad_token, 1)

    @staticmethod
    def build_vocab(texts: List[str], min_freq: int = 2, max_size: int = 50000) -> Dict[str, int]:
        from collections import Counter
        counter = Counter()
        for t in texts:
            tokens = SimpleTokenizer.tokenize_static(t)
            counter.update(tokens)
        # special tokens
        vocab = {"<unk>": 0, "<pad>": 1}
        for tok, freq in counter.most_common():
            if freq < min_freq:
                continue
            if tok in vocab:
                continue
            vocab[tok] = len(vocab)
            if len(vocab) >= max_size:
                break
        return vocab

    @staticmethod
    def tokenize_static(text: str) -> List[str]:
        return text.lower().strip().split()

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        tokens = self.tokenize_static(text)[:max_len]
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        return ids

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T]

# Transformer encoder classifier
class LocalTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1, pad_id: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.posenc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layernorm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T], attention_mask: [B, T] (1 for real, 0 for pad)
        x = self.embed(input_ids)  # [B, T, D]
        x = self.posenc(x)
        # Create key_padding_mask: True for pad positions
        key_padding_mask = attention_mask == 0  # [B, T]
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        # Pooling: mean over valid tokens
        mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        summed = (x * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        logits = self.classifier(pooled)
        return logits

# Dataset loader for CSV
INTENT_LIST = [
    "cari_rekomendasi_paket","estimasi_budget","cari_venue","tanya_kemungkinan","cari_dekor","cari_vendor","cari_catering"
]
INTENT_TO_ID = {label: i for i, label in enumerate(INTENT_LIST)}

class WeddingCSV(Dataset):
    def __init__(self, csv_path: str, tokenizer: SimpleTokenizer, max_len: int = 64):
        self.rows = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r['text']
        intent = r['intent']
        y = INTENT_TO_ID.get(intent, 0)
        ids = self.tokenizer.encode(text, max_len=self.max_len)
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(y, dtype=torch.long)
        }

# Collate with padding
def collate_fn(batch: List[Dict], pad_id: int = 1):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        ids = x['input_ids']
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
        input_ids.append(ids)
        attention_mask.append((ids != pad_id).long())
        labels.append(x['label'])
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }

# Training loop
def train_model(csv_path: str, save_dir: str = "models/local_transformer_intent", epochs: int = 2, batch_size: int = 64, lr: float = 3e-4, max_len: int = 64, device: str = None):
    # Build tokenizer vocab from CSV texts
    texts = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            texts.append(r['text'])
    vocab = SimpleTokenizer.build_vocab(texts, min_freq=2, max_size=50000)
    tokenizer = SimpleTokenizer(vocab)

    dataset = WeddingCSV(csv_path, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LocalTransformerClassifier(vocab_size=len(vocab), num_labels=len(INTENT_LIST), pad_id=tokenizer.pad_id)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += input_ids.size(0)
        avg_loss = total_loss / total_examples
        acc = total_correct / total_examples
        print(f"Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.4f}")

    # Save artifacts
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'vocab': vocab}, os.path.join(save_dir, 'model.pt'))
    with open(os.path.join(save_dir, 'intent_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(INTENT_LIST, f, ensure_ascii=False)
    print(f"Saved model and vocab to {save_dir}")

# Inference wrapper
class LocalIntentPipeline:
    def __init__(self, model_dir: str = "models/local_transformer_intent", device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        ckpt = torch.load(os.path.join(model_dir, 'model.pt'), map_location=device)
        self.vocab = ckpt['vocab']
        self.tokenizer = SimpleTokenizer(self.vocab)
        self.model = LocalTransformerClassifier(vocab_size=len(self.vocab), num_labels=len(INTENT_LIST), pad_id=self.tokenizer.pad_id)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def extract_slots_by_rule(text: str) -> Dict:
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

        # Tema (keyword sederhana; perluas sedikit)
        tema_keywords = [
            "sunda","jawa","rustic","internasional","international","minimalis","modern",
            "minang","batak","bugis","melayu","aceh","bali","betawi","madura",
            "dayak","toraja","banjar","sasak","ambon","maluku","papua","karo","nias"
        ]
        for k in tema_keywords:
            if k in lower:
                slots["tema"] = k if k != "international" else "internasional"
                break

        # Lokasi
        lokasi_list = [
            "bandung","jakarta pusat","jakarta barat","jakarta timur","jakarta selatan","jakarta utara","jakarta",
            "bogor","depok","bekasi","tangerang","tangerang selatan","bali","surabaya","malang","garut","cirebon",
            "tasikmalaya","pekanbaru","medan","makassar","yogya","yogyakarta","semarang","solo","surakarta","padang",
            "ambon","pontianak","banjarmasin","balikpapan","samarinda","manado","kendari","gorontalo","lombok","mataram",
            "ubud","denpasar","canggu","nusa dua","labuan bajo","kupang","jayapura","puncak","sentul"
        ]
        for city in lokasi_list:
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
        elif "siraman" in lower:
            slots["tipe_acara"] = "siraman"
        elif "midodareni" in lower:
            slots["tipe_acara"] = "midodareni"
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
        elif "cafe" in lower or "kafe" in lower:
            slots["venue"] = "cafe"
        elif "restoran" in lower:
            slots["venue"] = "restoran"
        elif "outdoor" in lower or "garden" in lower or "taman" in lower or "kebun" in lower:
            slots["venue"] = "outdoor"
        elif "ballroom" in lower:
            slots["venue"] = "ballroom"
        elif "rooftop" in lower:
            slots["venue"] = "rooftop"

        # Waktu
        if "malam" in lower:
            slots["waktu"] = "malam"
        elif "pagi" in lower:
            slots["waktu"] = "pagi"
        elif "siang" in lower:
            slots["waktu"] = "siang"
        elif "sore" in lower:
            slots["waktu"] = "sore"
        elif "minggu" in lower:
            slots["waktu"] = "minggu"
        elif "weekday" in lower:
            slots["waktu"] = "weekday"
        elif "weekend" in lower:
            slots["waktu"] = "weekend"

        # Budget (ambil pola angka juta)
        import re
        match_juta = re.findall(r"(\d+)\s*juta", lower)
        # juga tangkap pola seperti 60juta tanpa spasi
        match_juta_nospace = re.findall(r"(\d+)juta", lower)
        amounts = [int(x) * 1_000_000 for x in (match_juta + match_juta_nospace)]
        if amounts:
            slots["budget_min"] = min(amounts)
            slots["budget_max"] = max(amounts)

        # Jumlah tamu
        match_tamu = re.search(r"(\d+)\s*(orang|tamu|undangan|pax)", lower)
        if match_tamu:
            try:
                slots["jumlah_tamu"] = int(match_tamu.group(1))
            except ValueError:
                pass

        return slots

    def predict(self, text: str, max_len: int = 64) -> Dict:
        ids = self.tokenizer.encode(text, max_len=max_len)
        ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        attn = (ids_t != self.tokenizer.pad_id).long()
        with torch.no_grad():
            logits = self.model(ids_t.to(self.device), attn.to(self.device))
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_id = int(torch.argmax(logits, dim=-1)[0].cpu())
        slots = self.extract_slots_by_rule(text)
        return {
            'text': text,
            'intent_pred': INTENT_LIST[pred_id],
            'probs': {label: float(probs[i]) for i, label in enumerate(INTENT_LIST)},
            'slots': slots
        }

# CLI helpers
def cli_train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='dataset_pertanyaan_wedding.csv')
    parser.add_argument('--save_dir', default='models/local_transformer_intent')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_len', type=int, default=64)
    args = parser.parse_args()
    train_model(args.csv, args.save_dir, args.epochs, args.batch_size, args.lr, args.max_len)

def cli_infer():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='models/local_transformer_intent')
    args = parser.parse_args()
    pipe = LocalIntentPipeline(args.model_dir)
    print("Ketik pertanyaan (ketik 'exit' untuk keluar):")
    while True:
        try:
            query = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {'exit','quit'}:
            break
        out = pipe.predict(query)
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Local Transformer Intent Model')
    parser.add_argument('--mode', choices=['train','infer'], default='infer')
    parser.add_argument('--csv', default='dataset_pertanyaan_wedding.csv')
    parser.add_argument('--save_dir', default='models/local_transformer_intent')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--model_dir', default='models/local_transformer_intent')
    args = parser.parse_args()
    if args.mode == 'train':
        train_model(args.csv, args.save_dir, args.epochs, args.batch_size, args.lr, args.max_len)
    else:
        pipe = LocalIntentPipeline(args.model_dir)
        print("Ketik pertanyaan (ketik 'exit' untuk keluar):")
        while True:
            try:
                query = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in {'exit','quit'}:
                break
            out = pipe.predict(query)
            print(json.dumps(out, ensure_ascii=False, indent=2))
