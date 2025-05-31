import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0, 'dim must be divisible by heads'
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=True)  # Added bias
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, d = x.shape
        q = self.to_q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)  # (b, heads, n, head_dim)
        k = self.to_k(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, heads, n, n)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (b, heads, n, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, n, d)  # (b, n, d)
        out = self.to_out(out)
        out = self.dropout(out)  # Added output dropout
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
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        # Pre-norm + residual
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x)
        # Pre-norm + residual for FF
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x

class RowWiseTransformers(nn.Module):
    def __init__(self, num_features, dim, depth, heads, mlp_dim, dropout=0.2, row_norm=True):
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        self.row_norm = row_norm
        
        # Feature embedding
        self.feature_embed = nn.Linear(1, dim)
        
        # Learnable positional embeddings with Xavier init
        self.pos_embed = nn.Parameter(torch.zeros(num_features, dim))
        nn.init.xavier_uniform_(self.pos_embed)
        
        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        
        # Attention-based pooling
        self.pool_weight = nn.Parameter(torch.randn(dim, 1))
        self.to_out = nn.Linear(dim, 1)
    
    def forward(self, x):
        # x: (batch_size, num_features)
        # Clip inputs to prevent extreme values
        x = torch.clamp(x, -1e4, 1e4)
        
        # Row-wise normalization
        if self.row_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x_std = x.std(dim=-1, keepdim=True) + 1e-8
            x = (x - x_mean) / x_std
        
        # Embed features
        x = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        x = self.feature_embed(x)  # (batch_size, num_features, dim)
        
        # Add positional embeddings
        x = x + self.pos_embed.unsqueeze(0)  # (batch_size, num_features, dim)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Attention-based pooling
        weights = torch.softmax(x @ self.pool_weight, dim=1)  # (batch_size, num_features, 1)
        x = (x * weights).sum(dim=1)  # (batch_size, dim)
        
        # Output
        out = self.to_out(x)  # (batch_size, 1)
        return out.squeeze(-1)  # (batch_size,)

def build_transformers(num_features, dim, heads, depth, mlp_dim, dropout):
    return RowWiseTransformers(num_features, dim, depth, heads, mlp_dim, dropout)

if __name__ == "__main__":
    batch_size = 16
    num_features = 200
    model = build_transformers(num_features, dim=32, heads=4, depth=3, mlp_dim=128, dropout=0.2)
    dummy_x = torch.randn(batch_size, num_features)
    preds = model(dummy_x)
    print(preds.shape)  # Expected: (batch_size,)