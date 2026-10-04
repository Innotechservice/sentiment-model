import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "az_reviews.csv"
OUT_DIR = BASE_DIR / "data" / "processed"
URL = (
    "https://huggingface.co/datasets/hajili/"
    "azerbaijani_review_sentiment_classification/resolve/main/train.csv"
)


def az_clean(text):
    """Azerbaycan dilina uygun temizleme."""
    text = str(text)
    # Python-un standart kicik herf cevirmesi I ve i herflerini yanlis cevirir
    text = text.replace("\u0130", "i").replace("I", "\u0131")
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH, encoding="utf-8")
    else:
        print("Yuklenir...")
        df = pd.read_csv(URL)
        df.to_csv(RAW_PATH, index=False, encoding="utf-8")
    print("Sutunlar:", list(df.columns), "| setir:", len(df))

    # Metn sutunu: score-dan basqa, orta uzunlugu en boyuk sutun
    candidates = [c for c in df.columns if c != "score"]
    text_col = max(candidates, key=lambda c: df[c].astype(str).str.len().mean())
    print("Metn sutunu:", text_col)

    df = df[[text_col, "score"]].rename(columns={text_col: "text"}).dropna()
    df["text"] = df["text"].apply(az_clean)
    df = df[df["text"].str.len() > 0]

    before = len(df)
    df = df.drop_duplicates(subset="text")
    print("Tekrar edilen metnler silindi:", before - len(df))

    df = df[df["score"].isin([1, 2, 4, 5])].copy()
    df["label"] = (df["score"] >= 4).astype(int)
    print("3 ulduz silindikden sonra: menfi", int((df["label"] == 0).sum()),
          "| musbet", int((df["label"] == 1).sum()))

    n = int(df["label"].value_counts().min())
    parts = [g.sample(n, random_state=42) for _, g in df.groupby("label")]
    df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    print("Balanslasdirandan sonra: her sinifden", n, "| cemi", len(df))

    words = df["text"].str.split().str.len()
    print("Soz sayi: median", int(words.median()),
          "| 90 faiz", int(words.quantile(0.9)))

    train_df, rest = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        rest, test_size=0.5, random_state=42, stratify=rest["label"]
    )
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part[["text", "label"]].to_csv(
            OUT_DIR / f"az_{name}.csv", index=False, encoding="utf-8"
        )
        print(f"az_{name}.csv:", len(part), "setir")

    with open(OUT_DIR / "az_check.txt", "w", encoding="utf-8-sig") as f:
        for label, title in [(0, "MENFI"), (1, "MUSBET")]:
            f.write(f"===== {title} =====\n\n")
            for t in df[df["label"] == label]["text"].head(10):
                f.write(t[:300] + "\n\n")
    print("Yoxlama ucun data/processed/az_check.txt faylina baxin.")


if __name__ == "__main__":
    main()
