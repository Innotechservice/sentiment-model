import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD_ID = 0  # bos yer ucun nomre (vocab.py ile eyni)


class SentimentLSTM(nn.Module):
    """Yavas, amma soz sirasini nezere alan model."""

    def __init__(
        self,
        vocab_size=20000,
        embed_dim=128,
        hidden_dim=128,
        num_classes=2,
        dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lengths = (x != PAD_ID).sum(dim=1).clamp(min=1).cpu()

        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        features = torch.cat([hidden[0], hidden[1]], dim=1)
        features = self.dropout(features)
        return self.fc(features)


class SentimentBag(nn.Module):
    """Suretli model: soz vektorlarinin ortalamasi + kicik sinir sebekesi."""

    def __init__(
        self,
        vocab_size=20000,
        embed_dim=64,
        hidden_dim=64,
        num_classes=2,
        dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # mask: real soz = 1, bos yer = 0
        mask = (x != PAD_ID).unsqueeze(-1).float()
        emb = self.embedding(x) * mask
        counts = mask.sum(dim=1).clamp(min=1)
        mean = emb.sum(dim=1) / counts

        hidden = self.dropout(self.relu(self.fc1(mean)))
        return self.fc2(hidden)


if __name__ == "__main__":
    # Test ucun uydurma 4 reylik paket (hec bir real mena dasimir)
    x = torch.randint(2, 20000, (4, 256))
    x[0, 100:] = 0
    x[1, 50:] = 0

    for name, model in [("LSTM", SentimentLSTM()), ("Bag", SentimentBag())]:
        total = sum(p.numel() for p in model.parameters())
        out = model(x)
        print(name, "| parametr sayi:", total, "| cixis formasi:", tuple(out.shape))
