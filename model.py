import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states, mask=None):
        attn_weights = self.score(hidden_states).squeeze(-1)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attn_weights


class BiLSTMAttentionMCQ(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0)

        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def encode_one(self, x):
        mask = (x != 0)
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        ctx, _ = self.attention(out, mask)
        return ctx

    def forward(self, input_ids):
        B, N, L = input_ids.shape
        flat = input_ids.view(B * N, L)

        ctx = self.encode_one(flat)
        ctx = self.dropout(ctx)

        logits = self.classifier(ctx).squeeze(-1)
        logits = logits.view(B, N)

        return logits