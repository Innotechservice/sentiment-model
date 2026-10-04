import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SentimentBag
from train import run_epoch
from vocab import MODELS_DIR, PROCESSED_DIR, ReviewDataset, build_vocab, save_vocab

MAX_LEN = 64
MAX_VOCAB = 20000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
PATIENCE = 3
VOCAB_PATH = MODELS_DIR / "az_vocab.json"
MODEL_PATH = MODELS_DIR / "az_bag.pt"


def main():
    # Istifade: python src\az_train.py 20   (maksimum epoch sayi)
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    torch.manual_seed(42)

    train_path = PROCESSED_DIR / "az_train.csv"
    train_texts = (
        pd.read_csv(train_path, encoding="utf-8")["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    word2idx = build_vocab(train_texts, MAX_VOCAB)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_vocab(word2idx, VOCAB_PATH)
    print("Sozluk olcusu:", len(word2idx))

    train_data = ReviewDataset(train_path, word2idx, MAX_LEN)
    val_data = ReviewDataset(PROCESSED_DIR / "az_val.csv", word2idx, MAX_LEN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = SentimentBag(vocab_size=len(word2idx))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    print(f"Oyrenme: {len(train_data)} numune | yoxlama: {len(val_data)} numune")

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        seconds = int(time.time() - start)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | {seconds} san"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("  Yeni en yaxsi model saxlanildi.")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  Val loss {PATIENCE} epoch yaxsilasmadi, dayandirildi.")
                break

    print(
        f"YEKUN: en yaxsi epoch {best_epoch} | "
        f"val loss {best_val_loss:.4f} | val acc {best_val_acc:.3f}"
    )


if __name__ == "__main__":
    main()
