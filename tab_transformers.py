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

        # linear layers to generate Q, K, and V

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # final linear projection
        self.to_out = nn.Linear(dim, dim)  

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x -> (batch_seq, num_features, dim)
        # attention is over features

        b, n, d = x.shape # n is the num features

        # compute queries, keys, and values

        q = self.to_q(x) # (b, n, d)
        k = self.to_k(x) # (b, n, d)
        v = self.to_v(x) # (b, n, d)

        # split head (multi head attention)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1,2) # (b, heads, n, head_dim)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1,2) # (b, heads, n, head_dim)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1,2)

        # scaled dot product attention

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (b, heads, n, n)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        # lets combine heads now
        out = out.transpose(1, 2).contiguous().view(b, n, d) # (b, n, d)

        # final linear projection
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
        # attention + residual + norm
        x = x + self.attn(self.norm1(x))
        
        # feedforward + residual + norm
        x = x + self.ff(self.norm2(x))

        return x
        

class RowWiseTransformers(nn.Module):
    def __init__(self, num_features, dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        self.num_features = num_features
        self.dim = dim

        # embed each scalar feature to vector dim
        self.feature_embed = nn.Linear(1, dim)

        # positional embeddings per features
        self.pos_embed = nn.Parameter(torch.randn(1, num_features, dim))

        # stacks of transformers blocks
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        
        # final mlp head
        self.to_out = nn.Sequential(nn.Flatten(),
                                    nn.LayerNorm(num_features * dim),
                                    nn.Linear(num_features * dim, 1)
                                      )
        
    def forward(self, x):
        # x -> (batch, num_features)
        x = x.unsqueeze(-1) # (batch, num_features, 1)
        x = self.feature_embed(x) # (batch, num_features, dim)
        x = x + self.pos_embed # add positional encodings

        for layer in self.layers:
            x = layer(x)

        out = self.to_out(x) # (batch, 1)
        return out.squeeze(-1) # (batch,)
        

if __name__ == "__main__":
    batch_size = 16
    num_features = 200
    
    model = RowWiseTransformers(num_features=num_features, dim=64, heads = 4, depth=6, mlp_dim=2048, dropout=0.1)
    dummy_x = torch.randn(batch_size, num_features)

    preds = model(dummy_x)
    print(preds.shape)  # (batch_size,)











