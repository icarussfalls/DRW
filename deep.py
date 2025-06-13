import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class RobustFeatureProbabilityGate(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # More robust normalization with learnable affine parameters
        self.norm = nn.BatchNorm1d(num_features)
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.scale = nn.Parameter(torch.ones(num_features))
        
    def forward(self, x):
        x = self.norm(x)
        return torch.sigmoid(x * self.scale + self.bias)

class DriftAwareMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0, 'dim must be divisible by heads'
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        # Additional drift-resistant parameters
        self.attention_dropout = nn.Dropout(dropout)
        self.rescale_factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        b, n, d = x.shape
        
        # Combined QKV projection
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, self.head_dim).transpose(1, 2), qkv)
        
        # Scaled dot-product attention with rescaling
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.rescale_factor / math.sqrt(self.head_dim))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attention_dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.to_out(out)

class DriftRobustFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # Additional layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.rescale = nn.Parameter(torch.tensor(0.5))  # Learned residual scale

    def forward(self, x):
        return x * self.rescale + self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.attn = DriftAwareMultiHeadAttention(dim=dim, heads=heads, dropout=dropout)
        self.ln1 = nn.BatchNorm1d(dim)  # Using BatchNorm for better drift handling
        
        self.ff = DriftRobustFeedForward(dim, mlp_dim, dropout)
        self.ln2 = nn.BatchNorm1d(dim)
        
        # Additional stabilization
        self.attn_gate = nn.Parameter(torch.tensor(0.5))
        self.ff_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Permute for BatchNorm
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        
        # Attention path
        attn_out = self.attn(self.ln1(x).view(orig_shape))
        x = x.view(orig_shape) + attn_out * self.attn_gate
        
        # Feedforward path
        x_flat = x.reshape(-1, orig_shape[-1])
        ff_out = self.ff(self.ln2(x_flat)).view(orig_shape)
        return x + ff_out * self.ff_gate

class RobustRowWiseTransformers(nn.Module):
    def __init__(self, num_features, dim, depth, heads, mlp_dim, dropout, hidden_dim=64):
        super().__init__()
        
        # Enhanced feature processing
        self.input_net = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout * 2),  # Higher dropout for input
        )
        
        self.feature_gate = RobustFeatureProbabilityGate(num_features)
        
        # More powerful feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Learnable position embeddings (though data is shuffled)
        self.pos_embed = nn.Parameter(torch.randn(1, num_features, dim) * 0.02)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) 
            for _ in range(depth)
        ])
        
        # Robust output network
        self.to_out = nn.Sequential(
            nn.BatchNorm1d(num_features * dim),
            nn.Dropout(dropout),
            nn.Linear(num_features * dim, num_features * dim // 2),
            nn.GELU(),
            nn.LayerNorm(num_features * dim // 2),
            nn.Linear(num_features * dim // 2, 1)
        )
        
        # Initialization
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Input processing
        x = self.input_net(x)
        
        # Feature gating
        feature_weights = self.feature_gate(x)
        x = x * feature_weights
        
        # Feature embedding
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)
        x = x + self.pos_embed
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
            
        # Output processing
        x = x.flatten(1)
        return self.to_out(x).squeeze(-1)

def build_robust_transformers(num_features, dim=64, heads=4, depth=6, mlp_dim=256, dropout=0.2):
    return RobustRowWiseTransformers(
        num_features=num_features,
        dim=dim,
        heads=heads,
        depth=depth,
        mlp_dim=mlp_dim,
        dropout=dropout
    )

class DataAugmentation:
    """Augmentation to simulate test-time conditions"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        if torch.rand(1) < self.p:
            # Feature noise
            x = x * torch.randn_like(x).clamp(-0.3, 0.3) + 0.1 * torch.randn(1, device=x.device)
            
            # Random feature mixing
            if x.size(0) > 1 and torch.rand(1) < 0.3:
                idx = torch.randperm(x.size(0))
                lam = torch.rand(1, device=x.device) * 0.4 + 0.3
                x = lam * x + (1 - lam) * x[idx]
                
        return x

if __name__ == "__main__":
    batch_size = 32
    num_features = 200
    
    model = build_robust_transformers(
        num_features=num_features,
        dim=64,
        heads=4,
        depth=6,
        mlp_dim=256,
        dropout=0.2
    )
    
    dummy_x = torch.randn(batch_size, num_features)
    aug = DataAugmentation()
    
    # Test augmentation
    augmented_x = aug(dummy_x.clone())
    
    # Test model
    preds = model(dummy_x)
    print(f"Input shape: {dummy_x.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")