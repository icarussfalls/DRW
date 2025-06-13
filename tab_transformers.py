import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0, 'dim must be divisible by heads'
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.to_out(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(dim=dim, heads=heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class FeatureProbabilityGate(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.ln = nn.LayerNorm(num_features)
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x_norm = self.ln(x) / 3.0  # Stabilize sigmoid input
        probs = torch.sigmoid(x_norm + self.bias)
        return probs


class RowWiseTransformers(nn.Module):
    def __init__(self, num_features, dim, depth, heads, mlp_dim, dropout, hidden_dim=32):
        super().__init__()

        self.num_features = num_features
        self.dim = dim
        self.feature_gate = FeatureProbabilityGate(num_features)

        self.feature_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, num_features, dim) * 0.01)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(dim)

        self.to_out = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(num_features * dim),
            nn.Linear(num_features * dim, 1)
        )

    def forward(self, x):
        feature_weights = self.feature_gate(x)
        x = x * feature_weights

        x = x.unsqueeze(-1)
        x = self.feature_embed(x)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        out = self.to_out(x)
        return out.squeeze(-1)


def build_robust_transformers(num_features, dim, heads, depth, mlp_dim, dropout):
    return RowWiseTransformers(
        num_features=num_features,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout
    )


if __name__ == "__main__":
    batch_size = 16
    num_features = 200

    model = build_robust_transformers(
        num_features=num_features,
        dim=32,
        heads=2,
        depth=2,
        mlp_dim=128,
        dropout=0.1
    )

    dummy_x = torch.randn(batch_size, num_features)
    preds = model(dummy_x)
    print(preds.shape)  # Expected: (batch_size,)
