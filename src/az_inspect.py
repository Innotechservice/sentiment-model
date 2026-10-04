import pandas as pd

URL = (
    "https://huggingface.co/datasets/hajili/"
    "azerbaijani_review_sentiment_classification/resolve/main/train.csv"
)

df = pd.read_csv(URL)
print("Sutunlar:", list(df.columns))
print("Setir sayi:", len(df))

for col in df.columns:
    if df[col].nunique() <= 10:
        print("Sutun", col, "- say:")
        print(df[col].value_counts())

df.sample(30, random_state=1).to_csv(
    "data/processed/az_sample.csv", index=False, encoding="utf-8-sig"
)
print("30 numune data/processed/az_sample.csv faylina yazildi.")
