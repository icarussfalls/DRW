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
        b, n, d = x.shape  # n is the num features
        q = self.to_q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)  # (b, heads, n, head_dim)
        k = self.to_k(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, heads, n, n)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)  # (b, n, d)
        out = self.to_out(out)
        return out


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
    def __init__(self, dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class RowWiseTransformers(nn.Module):
    def __init__(self, num_features, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.num_features = num_features
        self.dim = dim

        # Feature normalization
        self.feature_norm = nn.BatchNorm1d(num_features)
        # Embed scalar features to vector dim
        self.feature_embed = nn.Linear(1, dim)
        # Optional: Interaction layer for initial feature interactions
        self.interaction_layer = nn.Sequential(
            nn.Linear(num_features, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        # Enhanced output head
        self.to_out = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(num_features * dim),
            nn.Linear(num_features * dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, x):
        # x: (batch, num_features)
        x_norm = self.feature_norm(x)  # Normalize features
        x = x_norm.unsqueeze(-1)  # (batch, num_features, 1)
        x = self.feature_embed(x)  # (batch, num_features, dim)
        # Add interaction layer output
        interaction_out = self.interaction_layer(x_norm).unsqueeze(1).expand(-1, self.num_features, -1)
        x = x + interaction_out
        for layer in self.layers:
            x = layer(x)
        out = self.to_out(x)  # (batch, 1)
        return out.squeeze(-1)  # (batch,)


def build_transformers(num_features, dim, heads, depth, mlp_dim, dropout):
    return RowWiseTransformers(num_features=num_features, dim=dim, heads=heads, depth=depth, mlp_dim=mlp_dim, dropout=dropout)


# Data augmentation for shuffled features
def permute_features(x):
    batch_size, num_features = x.shape
    perm = torch.randperm(num_features)
    return x[:, perm]


if __name__ == "__main__":
    batch_size = 16
    num_features = 200
    model = build_transformers(num_features=num_features, dim=64, heads=4, depth=6, mlp_dim=2048, dropout=0.2)
    dummy_x = torch.randn(batch_size, num_features)
    # Apply permutation augmentation
    dummy_x = permute_features(dummy_x)
    preds = model(dummy_x)
    print(preds.shape)  # (batch_size,)