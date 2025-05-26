import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, n_heads, seq_len, head_dim)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)  # (batch, heads, seq_len, head_dim)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Learnable projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Linear projections and split into heads
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Now Q,K,V are (batch_size, n_heads, seq_len, head_dim)

        out, attn = self.attention(Q, K, V, mask)  # out shape same as Q,K,V

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear layer
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
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + Add & Norm
        attn_out = self.mha(x, mask)
        x = self.norm1(x + attn_out)

        # Feed Forward + Add & Norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class Transformer(nn.Module):
    def __init__(self, num_features, d_model, n_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        # Project each scalar feature to d_model embedding
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_features)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output layer - output single regression value per sample
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch_size, num_features)
        batch_size, seq_len = x.size()
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        # Pool over features: mean
        x = x.mean(dim=1)  # (batch_size, d_model)

        out = self.fc_out(x).squeeze(-1)  # (batch_size,)
        return out

def build_transformers(num_features, d_model, n_heads, num_layers, d_ff, dropout):
    model = Transformer(num_features, d_model=d_model, n_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
    return model

# Test the model
if __name__ == "__main__":
    batch_size = 8
    num_features = 30

    model = Transformer(num_features=num_features, d_model=64, n_heads=8, num_layers=3, d_ff=256, dropout=0.1)
    sample_input = torch.randn(batch_size, num_features)

    output = model(sample_input)
    print("Output shape:", output.shape)  # Should be (batch_size,)
