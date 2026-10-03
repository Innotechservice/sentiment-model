import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SentimentBag, SentimentLSTM
from vocab import MODELS_DIR, PROCESSED_DIR, VOCAB_PATH, ReviewDataset, load_vocab

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01  # r?q?ml?rin boyumesini cezalandirir
PATIENCE = 3  # val loss bu qeder epoch yaxsilasmasa, dayanir
MODEL_CLASSES = {"bag": SentimentBag, "lstm": SentimentLSTM}


def run_epoch(model, loader, loss_fn, optimizer=None, verbose=False):
    """Bir epoch kecir. optimizer verilibse - oyrenir, verilmeyibse - yalniz yoxlayir."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for step, (x, y) in enumerate(loader, 1):
            logits = model(x)
            loss = loss_fn(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if verbose and step % 50 == 0:
                    print(f"  paket {step}/{len(loader)}, loss {loss.item():.4f}")

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total


def main():
    # Istifade: python src\train.py bag 20   (model adi, maksimum epoch sayi)
    name = sys.argv[1] if len(sys.argv) > 1 else "bag"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if name not in MODEL_CLASSES:
        print("Model adi 'bag' ve ya 'lstm' olmalidir.")
        return

    torch.manual_seed(42)
    model_path = MODELS_DIR / f"{name}.pt"

    word2idx = load_vocab(VOCAB_PATH)
    print("Data hazirlanir...")
    train_data = ReviewDataset(PROCESSED_DIR / "train.csv", word2idx)
    val_data = ReviewDataset(PROCESSED_DIR / "val.csv", word2idx)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = MODEL_CLASSES[name](vocab_size=len(word2idx))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    print(f"Model: {name}, maksimum epoch sayi: {epochs}")

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, optimizer, verbose=(name == "lstm")
        )
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
            torch.save(model.state_dict(), model_path)
            print("  Yeni en yaxsi model saxlanildi.")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  Val loss {PATIENCE} epoch yaxsilasmadi, training dayandirildi.")
                break

    print(
        f"YEKUN: en yaxsi epoch {best_epoch} | "
        f"val loss {best_val_loss:.4f} | val acc {best_val_acc:.3f}"
    )


if __name__ == "__main__":
    main()
