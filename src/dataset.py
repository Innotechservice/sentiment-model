import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE = RAW_DIR / "aclImdb_v1.tar.gz"
EXTRACTED = RAW_DIR / "aclImdb"


def download_imdb():
    """IMDB datasetini yukleyir ve acir."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACTED.exists():
        print("Dataset artiq var, tekrar yuklenmir.")
        return

    if not ARCHIVE.exists():
        print("Yuklenir (texminen 80 MB), gozleyin...")
        urllib.request.urlretrieve(URL, ARCHIVE)

    print("Arxiv acilir (bir nece deqiqe cekebilir)...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(RAW_DIR, filter="data")

    print("Hazirdir!")


def load_split(split):
    """split: 'train' ve ya 'test'. Cedvel qaytarir: text, label (1=musbet, 0=menfi)."""
    rows = []
    for folder_name, label in [("pos", 1), ("neg", 0)]:
        folder = EXTRACTED / split / folder_name
        for file in folder.glob("*.txt"):
            text = file.read_text(encoding="utf-8")
            rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


def clean_text(text):
    """Metni temizleyir: kicik herf, HTML isaresi yox, artiq bosluq yox."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data():
    """Datani temizleyir, train/validation/test hisselerine bolur ve saxlayir."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    full_train = load_split("train")
    test_df = load_split("test")

    full_train["text"] = full_train["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    train_df, val_df = train_test_split(
        full_train,
        test_size=0.2,
        random_state=42,
        stratify=full_train["label"],
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False, encoding="utf-8")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        musbet = int((df["label"] == 1).sum())
        menfi = int((df["label"] == 0).sum())
        print(f"{name}: {len(df)} setir (musbet {musbet}, menfi {menfi})")


if __name__ == "__main__":
    download_imdb()
    prepare_data()
