"""
models.py — Deep Learning Models untuk IMDB Sentiment Analysis (Binary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    NUM_CLASSES,
)

# =========================================================
# MODEL 1: BiLSTM
# =========================================================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,  # sekarang = 2
        dropout=DROPOUT,
        pad_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # concat forward + backward
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        out = self.dropout(hidden)
        logits = self.fc(out)

        return logits


# =========================================================
# MODEL 2: BiLSTM + Attention
# =========================================================
class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        pad_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Attention
        scores = torch.tanh(self.attention(output)).squeeze(-1)

        max_len = output.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=1)

        context = (attention_weights.unsqueeze(-1) * output).sum(dim=1)

        out = self.dropout(context)
        logits = self.fc(out)

        return logits, attention_weights

# =========================================================
# HELPER
# =========================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)