import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model import SentimentBag
from vocab import MODELS_DIR, PROCESSED_DIR, ReviewDataset, load_vocab, tokenize

MAX_LEN = 64
VOCAB_PATH = MODELS_DIR / "az_vocab.json"
MODEL_PATH = MODELS_DIR / "az_bag.pt"
NAMES = {0: "menfi", 1: "musbet"}


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
    word2idx = load_vocab(VOCAB_PATH)
    test_path = PROCESSED_DIR / "az_test.csv"
    test_data = ReviewDataset(test_path, word2idx, MAX_LEN)
    loader = DataLoader(test_data, batch_size=256)

    model = SentimentBag(vocab_size=len(word2idx))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)

    preds, confs = predict_all(model, loader)
    labels = test_data.labels.tolist()

    df = pd.read_csv(test_path, encoding="utf-8")
    texts = df["text"].fillna("").astype(str).tolist()

    # Test sozlerinin orta neche fayizi sozlukde var
    ratios = []
    for t in texts:
        words = tokenize(t)
        if words:
            ratios.append(sum(w in word2idx for w in words) / len(words))
    coverage = sum(ratios) / len(ratios)

    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=["menfi", "musbet"], digits=3
    )
    wrong = [i for i, (l, p) in enumerate(zip(labels, preds)) if l != p]
    wrong.sort(key=lambda i: -confs[i])

    summary = []
    summary.append(f"Test numune sayi: {len(labels)}")
    summary.append(f"ACCURACY: {accuracy_score(labels, preds):.4f}")
    summary.append(f"Test sozlerinin sozlukde olma payi (orta): {coverage:.1%}")
    summary.append("")
    summary.append(report)
    summary.append("Qarisiqliq cedveli (setir = real, sutun = model):")
    summary.append("              model:menfi  model:musbet")
    summary.append(f"real menfi    {cm[0][0]:>11}  {cm[0][1]:>12}")
    summary.append(f"real musbet   {cm[1][0]:>11}  {cm[1][1]:>12}")
    summary.append("")
    summary.append(f"Sehv sayi: {len(wrong)}")
    print("\n".join(summary))

    out_path = PROCESSED_DIR / "az_errors.txt"
    with open(out_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(summary))
        f.write("\n\n===== Modelin en emin oldugu 15 sehv =====\n\n")
        for i in wrong[:15]:
            f.write(
                f"real: {NAMES[labels[i]]} | model: {NAMES[preds[i]]} "
                f"({confs[i]:.0%} emin)\n"
            )
            f.write(texts[i][:300] + "\n\n")
    print("Sehv numuneleri yazildi: data/processed/az_errors.txt")


if __name__ == "__main__":
    main()