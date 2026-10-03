import sys

import torch

from dataset import clean_text
from model import SentimentBag
from vocab import MODELS_DIR, VOCAB_PATH, encode, load_vocab, tokenize

LABELS = {0: "menfi", 1: "musbet"}


class Predictor:
    """Oyredilmis modeli yukleyir ve yeni cumleleri proqnozlasdirir."""

    def __init__(self):
        self.word2idx = load_vocab(VOCAB_PATH)
        self.model = SentimentBag(vocab_size=len(self.word2idx))
        state = torch.load(MODELS_DIR / "bag.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, text):
        cleaned = clean_text(text)
        words = tokenize(cleaned)
        known = sum(1 for w in words if w in self.word2idx)

        ids = encode(cleaned, self.word2idx)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        label = int(probs.argmax())
        return {
            "label": LABELS[label],
            "confidence": float(probs[label]),
            "known_words": known,
            "total_words": len(words),
        }


def show(result):
    print(f"Netice: {result['label']} ({result['confidence']:.0%} emin)")
    print(f"Taninan sozler: {result['known_words']}/{result['total_words']}")
    total = result["total_words"]
    if total == 0:
        print("Diqqet: cumlede soz tapilmadi.")
    elif result["known_words"] / total < 0.5:
        print("Diqqet: sozlerin yarisindan coxu modele tanis deyil, netice etibarsiz ola biler.")


def main():
    predictor = Predictor()

    # Istifade 1: python src\predict.py "cumle"
    if len(sys.argv) > 1:
        show(predictor.predict(" ".join(sys.argv[1:])))
        return

    # Istifade 2: python src\predict.py  (canli rejim)
    print("Cumle yazin (cixmaq ucun bos buraxib Enter basin).")
    while True:
        text = input("Cumle: ").strip()
        if not text:
            break
        show(predictor.predict(text))
        print()


if __name__ == "__main__":
    main()
