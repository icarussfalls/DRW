import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

# Scaled Dot-Product Attention with better numerical stability
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout, temperature=1.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature
    
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(d_k) * self.temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Add small epsilon for numerical stability
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn

# Multi-Head Attention with improved regularization
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, headdrop):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.headdrop = headdrop

        # Use smaller initialization for better stability
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout, temperature=1.2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for attention outputs
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        out, attn = self.attention(Q, K, V, mask)

        # More conservative HeadDrop
        if self.training and self.headdrop > 0.0:
            # Keep at least 50% of heads
            effective_headdrop = min(self.headdrop, 0.5)
            drop_mask = (torch.rand(self.n_heads, device=x.device) > effective_headdrop).float()
            # Ensure at least one head survives
            if drop_mask.sum() == 0:
                drop_mask[0] = 1.0
            drop_mask = drop_mask.view(1, self.n_heads, 1, 1)
            out = out * drop_mask

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        return self.norm(out)

# Improved Feedforward with GELU activation
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        out = self.linear1(x)
        out = F.gelu(out)  # GELU instead of ReLU for smoother gradients
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return self.norm(out)

# Encoder Layer with Pre-Norm and more conservative LayerDrop
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, layerdrop, headdrop):
        super().__init__()
        self.layerdrop = min(layerdrop, 0.3)  # Cap layerdrop at 30%
        self.mha = MultiHeadAttention(d_model, n_heads, dropout, headdrop)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Skip entire layer only occasionally
        if self.training and random.random() < self.layerdrop:
            return x
            
        # Pre-norm architecture for better gradient flow
        normed_x = self.norm1(x)
        attn_out = self.mha(normed_x, mask)
        x = x + attn_out
        
        normed_x = self.norm2(x)
        ff_out = self.ff(normed_x)
        x = x + ff_out
        
        return x

# Sinusoidal + Learnable Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # Sinusoidal encoding for generalization
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Small learnable component
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        seq_len = x.size(1)
        sinusoidal = self.pe[:, :seq_len, :]
        learnable = self.learnable_pe[:, :seq_len, :]
        return x + sinusoidal + learnable

# Multi-scale Attention Pooling
class MultiScalePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.global_attn = nn.Parameter(torch.randn(d_model) * 0.02)
        self.local_conv = nn.Conv1d(d_model, d_model//4, kernel_size=3, padding=1)
        self.local_attn = nn.Parameter(torch.randn(d_model//4) * 0.02)
        self.combine = nn.Linear(d_model + d_model//4, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Global attention pooling
        global_scores = torch.matmul(x, self.global_attn)
        global_weights = torch.softmax(global_scores, dim=1).unsqueeze(-1)
        global_pooled = (x * global_weights).sum(dim=1)
        
        # Local feature extraction
        local_features = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        local_scores = torch.matmul(local_features, self.local_attn)
        local_weights = torch.softmax(local_scores, dim=1).unsqueeze(-1)
        local_pooled = (local_features * local_weights).sum(dim=1)
        
        # Combine features
        combined = torch.cat([global_pooled, local_pooled], dim=-1)
        output = self.combine(combined)
        return self.norm(output)

# Improved Model with better regularization
class ImprovedTransformer(nn.Module):
    def __init__(self, num_features, d_model, n_heads, num_layers, d_ff, dropout, layerdrop, headdrop):
        super().__init__()
        
        # Multi-scale input processing
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model//2),
            nn.GELU(),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_features)
        
        # Gradually decreasing regularization through layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_dropout = dropout * (0.5 + 0.5 * i / max(1, num_layers - 1))
            layer_layerdrop = layerdrop * (0.5 + 0.5 * i / max(1, num_layers - 1))
            layer_headdrop = headdrop * (0.5 + 0.5 * i / max(1, num_layers - 1))
            
            self.layers.append(
                TransformerEncoderLayer(d_model, n_heads, d_ff, 
                                      layer_dropout, layer_layerdrop, layer_headdrop)
            )
        
        self.final_norm = nn.LayerNorm(d_model)
        self.pooling = MultiScalePooling(d_model)
        
        # Output head with residual connection and regularization
        self.output_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1)
        )
        
        # Output scaling for numerical stability
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # Input processing
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.input_conv(x)  # (B, D, T)
        x = x.transpose(1, 2)  # (B, T, D)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Pooling and output
        pooled = self.pooling(x)
        output = self.output_head(pooled).squeeze(-1)
        
        # Scale output to prevent extreme predictions
        # output = output * torch.sigmoid(self.output_scale)
        
        return output

# Model builder with better initialization
def build_improved_transformer(num_features, d_model=64, n_heads=4, num_layers=3, 
                             d_ff=128, dropout=0.15, layerdrop=0.1, headdrop=0.1):
    """
    Conservative hyperparameters for better generalization:
    - Reduced model capacity (smaller d_model, d_ff)
    - Fewer heads to reduce overfitting
    - Higher dropout for regularization
    - Conservative layer/head drop rates
    """
    model = ImprovedTransformer(
        num_features=num_features,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        layerdrop=layerdrop,
        headdrop=headdrop
    )
    
    # Improved initialization
    for name, param in model.named_parameters():
        if param.dim() > 1:
            if 'conv' in name or 'linear' in name or 'out_proj' in name:
                nn.init.xavier_normal_(param, gain=0.8)  # Smaller gain
            elif 'attn_vector' in name or 'global_attn' in name or 'local_attn' in name:
                nn.init.normal_(param, std=0.02)
            else:
                nn.init.xavier_uniform_(param)
        elif param.dim() == 1 and 'bias' in name:
            nn.init.zeros_(param)
    
    return model

# Training utilities for better generalization
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float('inf')
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            self.best_model = model.state_dict().copy()
        else:
            self.wait += 1
        return self.wait >= self.patience

# Example usage with conservative settings for DRW competition
# if __name__ == "__main__":
#     # Even more conservative settings for financial data
#     model = build_improved_transformer(
#         num_features=30,    # Adjust based on your feature count
#         d_model=32,         # Smaller model
#         n_heads=2,          # Fewer heads
#         num_layers=2,       # Fewer layers
#         d_ff=64,           # Smaller FF dimension
#         dropout=0.2,        # Higher dropout
#         layerdrop=0.05,     # Very conservative layer drop
#         headdrop=0.05       # Very conservative head drop
#     )
    
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
#     sample_input = torch.randn(8, 30)
#     output = model(sample_input)
#     print(f"Output shape: {output.shape}")
#     print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

# if __name__ == "__main__":
#     model = build_transformer(
#         num_features=30,
#         d_model=64,
#         n_heads=8,
#         num_layers=3,
#         d_ff=256,
#         dropout=0.1,
#         layerdrop=0.1,
#         headdrop=0.1
#     )
#     sample_input = torch.randn(8, 30)  # batch_size=8, sequence length=30
#     output = model(sample_input)
#     print(output.shape)  # Expected: (8,)
