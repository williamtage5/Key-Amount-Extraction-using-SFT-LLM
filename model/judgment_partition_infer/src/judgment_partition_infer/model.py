from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .constants import DROPOUT, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, ZONE_ORDER


class PointerModel(nn.Module):
    """
    Sentence-level pointer model:
    - Encode each sentence by averaging character embeddings.
    - Encode sentence sequence with BiLSTM.
    - Use 6 independent heads to pick boundary sentence indices.
    """

    def __init__(self, vocab_size: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_id)
        self.dropout = nn.Dropout(DROPOUT)
        self.encoder = nn.LSTM(
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.heads = nn.ModuleList(
            [nn.Linear(HIDDEN_DIM * 2, 1) for _ in range(len(ZONE_ORDER) - 1)]
        )

    def forward(self, sent_chars: torch.Tensor, sent_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sent_chars: (B, S, C)
        # sent_mask: (B, S)
        emb = self.embedding(sent_chars)  # (B, S, C, D)
        mask = (sent_chars != self.pad_id).float()
        lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sent_vec = (emb * mask.unsqueeze(-1)).sum(dim=2) / lengths  # (B, S, D)
        sent_vec = self.dropout(sent_vec)
        encoded, _ = self.encoder(sent_vec)  # (B, S, 2H)
        encoded = self.dropout(encoded)

        logits = []
        for head in self.heads:
            logit = head(encoded).squeeze(-1)  # (B, S)
            logit = logit.masked_fill(~sent_mask, -1e9)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # (B, K, S)
        return logits, sent_vec

