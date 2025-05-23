import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].detach()
        return self.dropout(x)

# Input Embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        # For continuous inputs, use linear projection
        self.linear = nn.Linear(input_dim, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        return self.linear(x) * math.sqrt(self.d_model)


# Layer Normalization (custom implementation)
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * norm + self.bias

# Multi-Head Attention Block
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Query, Key, Value linear layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query)  # (batch, seq_len, d_model)
        K = self.w_k(key)
        V = self.w_v(value)

        # Split into multiple heads and transpose
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq_len, seq_len)

        if mask is not None:
            # mask shape: (batch, 1, 1, seq_len) or broadcastable
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (batch, heads, seq_len, d_k)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch, seq_len, d_model)

        # Final linear layer
        out = self.w_o(out)

        return out

# Feed Forward Block
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# Residual Connection + LayerNorm block
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Encoder Block (one Transformer encoder layer)
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout)
        ])

    def forward(self, x, src_mask=None):
        # Self attention + residual
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        # Feed forward + residual
        x = self.residual_connections[1](x, self.feed_forward)
        return x

# Encoder: stack of EncoderBlocks
class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

# the below approach introduces layer drop with probability, in which we drop a layer
# class Encoder(nn.Module):
#     def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1, layerdrop: float = 0.0):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
#         ])
#         self.norm = LayerNormalization(d_model)
#         self.layerdrop = layerdrop  # Probability of dropping a layer

#     def forward(self, x, src_mask=None):
#         for layer in self.layers:
#             if self.training and torch.rand(1).item() < self.layerdrop:
#                 continue  # Skip this layer
#             x = layer(x, src_mask)
#         return self.norm(x)

# Projection Layer to output vocab probabilities or regression output
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.proj(x)

# Full Transformer model (only encoder, for single step or sequence outputs)
class Transformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int, max_seq_len: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = InputEmbeddings(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.projection = ProjectionLayer(d_model, output_dim)

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        print("After embedding:", torch.isnan(x).any(), x.min().item(), x.max().item())
        
        x = self.positional_encoding(x)
        print("After positional encoding:", torch.isnan(x).any(), x.min().item(), x.max().item())
        
        x = self.encoder(x, src_mask)
        print("After encoder:", torch.isnan(x).any(), x.min().item(), x.max().item())
        
        out = self.projection(x)
        print("After projection:", torch.isnan(out).any(), out.min().item(), out.max().item())
        
        return out



# Function to build the transformer with typical hyperparameters
def build_transformer(input_dim, d_model, num_heads, d_ff,
                      num_layers, max_seq_len, output_dim, dropout):
    transformer = Transformer(input_dim, d_model, num_heads, d_ff, num_layers, max_seq_len, output_dim, dropout)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


# Example usage:
if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    input_dim = 900  # number of features per time step

    model = build_transformer(input_dim=input_dim, d_model=512, num_heads=8, d_ff=2048,
                              num_layers=6, max_seq_len=100, output_dim=1, dropout=0.1)

    src = torch.randn(batch_size, seq_len, input_dim)
    logits = model(src)  # shape: (batch_size, seq_len, output_dim)
    print(logits.shape)  # Expected: torch.Size([32, 60, 1])