import sys

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model import SentimentBag, SentimentLSTM
from vocab import MODELS_DIR, PROCESSED_DIR, VOCAB_PATH, ReviewDataset, load_vocab

MODEL_CLASSES = {"bag": SentimentBag, "lstm": SentimentLSTM}


def predict_all(model, loader):
    """Butun numuneler ucun proqnoz ve modelin emin oldugu faizi qaytarir."""
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for x, _ in loader:
            probs = torch.softmax(model(x), dim=1)
            conf, pred = probs.max(dim=1)
            preds.extend(pred.tolist())
            confs.extend(conf.tolist())
    return preds, confs


def main():
    # Istifade: python src\evaluate.py bag
    name = sys.argv[1] if len(sys.argv) > 1 else "bag"
    if name not in MODEL_CLASSES:
        print("Model adi 'bag' ve ya 'lstm' olmalidir.")
        return

    word2idx = load_vocab(VOCAB_PATH)
    test_path = PROCESSED_DIR / "test.csv"
    test_data = ReviewDataset(test_path, word2idx)
    loader = DataLoader(test_data, batch_size=256)

    model = MODEL_CLASSES[name](vocab_size=len(word2idx))
    state = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu")
    model.load_state_dict(state)

    preds, confs = predict_all(model, loader)
    labels = test_data.labels.tolist()

    print(f"Model: {name} | test numune sayi: {len(labels)}")
    print(f"ACCURACY: {accuracy_score(labels, preds):.4f}")
    print()
    print(classification_report(labels, preds, target_names=["menfi", "musbet"], digits=3))

    cm = confusion_matrix(labels, preds)
    print("Qarisiqliq cedveli (setir = real, sutun = model):")
    print("              model:menfi  model:musbet")
    print(f"real menfi    {cm[0][0]:>11}  {cm[0][1]:>12}")
    print(f"real musbet   {cm[1][0]:>11}  {cm[1][1]:>12}")

    df = pd.read_csv(test_path, encoding="utf-8")
    wrong = [i for i, (l, p) in enumerate(zip(labels, preds)) if l != p]
    wrong.sort(key=lambda i: -confs[i])
    names = {0: "menfi", 1: "musbet"}

    print()
    print(f"Sehv sayi: {len(wrong)}. Modelin en emin oldugu 3 sehv:")
    for i in wrong[:3]:
        text = str(df["text"][i])[:200].encode("ascii", "replace").decode()
        print(f"- real: {names[labels[i]]}, model: {names[preds[i]]} ({confs[i]:.0%} emin)")
        print(f"  {text}")


if __name__ == "__main__":
    main()
