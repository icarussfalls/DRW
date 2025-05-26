import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, headdrop):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.headdrop = headdrop

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        out, attn = self.attention(Q, K, V, mask)  # (B, H, T, D)

        # 🔥 HeadDrop: randomly zero out heads
        if self.training and self.headdrop > 0.0:
            drop_mask = (torch.rand(self.n_heads, device=x.device) > self.headdrop).float()
            drop_mask = drop_mask.view(1, self.n_heads, 1, 1)
            out = out * drop_mask  # apply head mask

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, layerdrop, headdrop):
        super().__init__()
        self.layerdrop = layerdrop
        self.mha = MultiHeadAttention(d_model, n_heads, dropout, headdrop)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.training and random.random() < self.layerdrop:
            # Skip this layer entirely during training
            return x
        attn_out = self.mha(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))  # Learnable!
    
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


class Pooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_vector = nn.Parameter(torch.randn(d_model))
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        scores = torch.matmul(x, self.attn_vector)  # (batch_size, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        pooled = (x * weights).sum(dim=1)
        return pooled

class TransformerWithConv(nn.Module):
    def __init__(self, num_features, d_model, n_heads, num_layers, d_ff, dropout, layerdrop, headdrop):
        super().__init__()
        # 1D convolution layer to extract local temporal features
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_features)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, layerdrop, headdrop) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pooling = Pooling(d_model)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len)
        x = self.conv1(x)   # (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        x = self.pooling(x)
        out = self.fc_out(x).squeeze(-1)
        return out

def build_transformer(num_features, d_model, n_heads, num_layers, d_ff, dropout, layerdrop, headdrop):
    transformer = TransformerWithConv(
        num_features,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        layerdrop=layerdrop,
        headdrop=headdrop
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer

# Example usage:
# model = build_transformer(num_features=30, d_model=64, n_heads=8, num_layers=3, d_ff=256, dropout=0.1, layerdrop=0.1, headdrop=0.1)
# sample_input = torch.randn(8, 30)
# output = model(sample_input)
# print(output.shape)  # (8,)
