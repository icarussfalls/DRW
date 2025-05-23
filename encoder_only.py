import torch
import torch.nn as nn
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Input Embedding
class InputEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.linear(x)

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.size()
        qkv = self.qkv_proj(x)
        Q, K, V = torch.chunk(qkv, 3, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, -1)

        return self.out_proj(context)

# Feedforward
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

# Full Encoder-only Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4, d_ff=256, num_layers=3):
        super().__init__()
        self.embedding = InputEmbedding(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        self.encoder = nn.Sequential(
            *[EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # Mean pooling over time
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)  # for pooling (B, D, T)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def test():
    model = TimeSeriesTransformer(input_dim=6)
    dummy_input = torch.randn(32, 30, 6)  # (B, T, D)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # (B, 1)

test()
