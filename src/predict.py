import re
import sys

import torch

from dataset import clean_text
from model import SentimentBag
from vocab import MAX_LEN, MODELS_DIR, VOCAB_PATH, encode, load_vocab, tokenize

LABELS = {0: "menfi", 1: "musbet"}


def az_clean(text):
    """az_data.py faylindaki az_clean funksiyasinin suretidir.

    Azerbaycan modeli bu temizleme ile oyredilib. Boyuk I ve i herflerini
    Azerbaycan qaydasi ile kicildir (Python-un standart lower() funksiyasi
    boyuk i herfini pozur).
    """
    text = str(text)
    text = text.replace("\u0130", "i").replace("I", "\u0131")
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Her dil ucun: model fayli, lugat fayli, cumle uzunlugu ve temizleme funksiyasi
LANGS = {
    "en": {
        "model": MODELS_DIR / "bag.pt",
        "vocab": VOCAB_PATH,
        "max_len": MAX_LEN,
        "clean": clean_text,
    },
    "az": {
        "model": MODELS_DIR / "az_bag.pt",
        "vocab": MODELS_DIR / "az_vocab.json",
        "max_len": 64,
        "clean": az_clean,
    },
}


class Predictor:
    """Oyredilmis modeli yukleyir ve yeni cumleleri proqnozlasdirir.

    lang="en" -> ingilis modeli (default), lang="az" -> azerbaycan modeli.
    """

    def __init__(self, lang="en"):
        if lang not in LANGS:
            raise ValueError(f"Namelum dil: {lang}. Mumkun: {list(LANGS)}")
        cfg = LANGS[lang]
        self.lang = lang
        self.max_len = cfg["max_len"]
        self.clean = cfg["clean"]
        self.word2idx = load_vocab(cfg["vocab"])
        self.model = SentimentBag(vocab_size=len(self.word2idx))
        state = torch.load(cfg["model"], map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, text):
        cleaned = self.clean(text)
        words = tokenize(cleaned)
        known = sum(1 for w in words if w in self.word2idx)

        ids = encode(cleaned, self.word2idx, self.max_len)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        label = int(probs.argmax())
        return {
            "label": LABELS[label],
            "confidence": float(probs[label]),
            "known_words": known,
            "total_words": len(words),
            "lang": self.lang,
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
    args = sys.argv[1:]
    lang = "en"
    if len(args) >= 2 and args[0] == "--lang":
        lang = args[1]
        args = args[2:]

    predictor = Predictor(lang)

    # Istifade 1: python src\predict.py --lang az "cumle"
    if args:
        show(predictor.predict(" ".join(args)))
        return

    # Istifade 2: python src\predict.py --lang az  (canli rejim)
    print(f"Dil: {lang}. Cumle yazin (cixmaq ucun bos buraxib Enter basin).")
    while True:
        text = input("Cumle: ").strip()
        if not text:
            break
        show(predictor.predict(text))
        print()


if __name__ == "__main__":
    main()