from tabular_transformers import *

class LearnableGaussianNoise(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # log sigma for numerical stability, one per feature
        self.log_sigma = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        if self.training:
            sigma = torch.exp(self.log_sigma)  # std dev > 0
            noise = torch.randn_like(x) * sigma
            return x + noise
        else:
            return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Add learnable noise layer here
        self.learnable_noise = LearnableGaussianNoise(d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        out, attn = self.attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        
        # Inject learnable noise here
        out = self.learnable_noise(out)

        out = self.dropout(out)
        return out
