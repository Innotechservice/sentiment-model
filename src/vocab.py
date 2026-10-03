import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
VOCAB_PATH = MODELS_DIR / "vocab.json"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

MAX_VOCAB = 20000
MAX_LEN = 256


def tokenize(text):
    """Metni sozlere bolur (kicik herflerle)."""
    return re.findall(r"\w+", text.lower())


def build_vocab(texts, max_vocab=MAX_VOCAB):
    """En cox islenen sozlere nomre verir: {soz: nomre}."""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    word2idx = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for word, _ in counter.most_common(max_vocab - 2):
        word2idx[word] = len(word2idx)
    return word2idx


def save_vocab(word2idx, path=VOCAB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)


def load_vocab(path=VOCAB_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode(text, word2idx, max_len=MAX_LEN):
    """Metni reqem siyahisina cevirir, uzunlugu max_len edir."""
    ids = [word2idx.get(word, UNK_ID) for word in tokenize(text)]
    ids = ids[:max_len]
    ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids


class ReviewDataset(Dataset):
    """CSV faylini PyTorch-un basa dusduyu formata salir."""

    def __init__(self, csv_path, word2idx, max_len=MAX_LEN):
        df = pd.read_csv(csv_path, encoding="utf-8")
        texts = df["text"].fillna("").astype(str).tolist()
        self.inputs = torch.tensor(
            [encode(t, word2idx, max_len) for t in texts], dtype=torch.long
        )
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


if __name__ == "__main__":
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", encoding="utf-8")
    train_texts = train_df["text"].fillna("").astype(str).tolist()

    lengths = pd.Series([len(tokenize(t)) for t in train_texts])
    print("Reylerin uzunlugu (soz sayi): median", int(lengths.median()), ", 90 faiz:", int(lengths.quantile(0.9)))
    print("MAX_LEN-den uzun reylerin payi:", round(float((lengths > MAX_LEN).mean()) * 100, 1), "faiz")

    word2idx = build_vocab(train_texts)
    save_vocab(word2idx)
    print("Sozluk olcusu:", len(word2idx))

    sample = "This movie is really good, but the ending was a bit slow"
    print("Numune cumle:", sample)
    print("Tokenler:", tokenize(sample))
    print("Reqemler (ilk 12):", encode(sample, word2idx)[:12])

    train_data = ReviewDataset(PROCESSED_DIR / "train.csv", word2idx)
    print("Dataset olcusu:", len(train_data))
    x, y = train_data[0]
    print("Bir numunenin formasi:", tuple(x.shape), "etiket:", int(y))
