# class Diffusion1D(nn.Module):
#     def __init__(self, num_heads, init_D=0.05, steps=1):
#         super().__init__()
#         # One learnable D per head (positive via exponentiation)
#         self.log_D = nn.Parameter(torch.log(torch.ones(num_heads) * init_D))
#         self.steps = steps

#     def forward(self, x):
#         # x: (B, H, N, D)
#         B, H, N, D = x.shape
#         for _ in range(self.steps):
#             x_padded = F.pad(x, (0, 0, 1, 1), mode='reflect')
#             laplacian = x_padded[:, :, 2:, :] - 2 * x + x_padded[:, :, :-2, :]
#             D_values = self.log_D.exp().view(1, H, 1, 1)  # shape (1, H, 1, 1)
#             x = x + D_values * laplacian
#         return x

# class HigherOrderDiffusion1D(nn.Module):
#     def __init__(self, num_heads, init_D2=0.05, init_D4=0.01, steps=1):
#         super().__init__()
#         self.log_D2 = nn.Parameter(torch.log(torch.ones(num_heads) * init_D2))
#         self.log_D4 = nn.Parameter(torch.log(torch.ones(num_heads) * init_D4))
#         self.steps = steps

#     def forward(self, x):
#         # x shape: (B, H, N, D)
#         B, H, N, D = x.shape
        
#         for _ in range(self.steps):
#             # Pad for 2nd order laplacian (1-neighbor)
#             x_padded_2 = F.pad(x, (0, 0, 1, 1), mode='reflect')
#             laplacian_2 = x_padded_2[:, :, 2:, :] - 2 * x + x_padded_2[:, :, :-2, :]
            
#             # Pad for 4th order laplacian (2-neighbors)
#             x_padded_4 = F.pad(x, (0, 0, 2, 2), mode='reflect')
#             laplacian_4 = (x_padded_4[:, :, :-4, :] - 4 * x_padded_4[:, :, 1:-3, :] + 
#                            6 * x - 4 * x_padded_4[:, :, 3:-1, :] + x_padded_4[:, :, 4:, :])
            
#             D2 = self.log_D2.exp().view(1, H, 1, 1)
#             D4 = self.log_D4.exp().view(1, H, 1, 1)
            
#             # Update with both 2nd and 4th order diffusion terms
#             x = x + D2 * laplacian_2 - D4 * laplacian_4
        
#         return x
    
# class DiffusedMultiheadSelfAttention(nn.Module):
#     def __init__(self, dim, heads=8, diffusion_steps=1):
#         super().__init__()
#         assert dim % heads == 0, "dim must be divisible by heads"
#         self.heads = heads
#         self.dim = dim
#         self.head_dim = dim // heads
#         self.scale = self.head_dim ** -0.5

#         self.qkv_proj = nn.Linear(dim, dim * 3)
#         self.out_proj = nn.Linear(dim, dim)

#         self.diffusion = Diffusion1D(num_heads=heads, init_D=0.05, steps=diffusion_steps)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv_proj(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, N, D)

#         # Apply explicit diffusion
#         q = self.diffusion(q)
#         k = self.diffusion(k)
#         v = self.diffusion(v)

#         # Scaled dot-product attention
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
#         attn = attn_scores.softmax(dim=-1)
#         out = torch.matmul(attn, v)  # (B, H, N, D)

#         out = out.transpose(1, 2).reshape(B, N, C)
#         return self.out_proj(out)

# class DualHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads, local_window, attn_dropout=0.1):
#         super().__init__()
#         assert num_heads % 2 == 0, "num_heads must be even"
#         self.num_heads = num_heads
#         self.d_model = d_model
#         self.head_dim = d_model // num_heads
#         self.local_window = local_window

#         self.qkv_proj = nn.Linear(d_model, 3 * d_model)
#         self.out_proj = nn.Linear(d_model, d_model)
#         self.attn_dropout = nn.Dropout(attn_dropout)

#     def forward(self, x, mask=None):
#         B, T, C = x.size()
#         qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
#         qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         Q, K, V = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, T, head_dim)

#         half = self.num_heads // 2
#         Q_global, K_global, V_global = Q[:, :half], K[:, :half], V[:, :half]
#         Q_local, K_local, V_local = Q[:, half:], K[:, half:], V[:, half:]

#         # Global Attention
#         scores_global = torch.matmul(Q_global, K_global.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             # mask shape should broadcast to scores_global shape: (B, half, T, T)
#             scores_global = scores_global.masked_fill(mask == 0, float('-inf'))
#         attn_global = self.attn_dropout(F.softmax(scores_global, dim=-1))
#         out_global = torch.matmul(attn_global, V_global)

#         # Local Attention
#         scores_local = torch.matmul(Q_local, K_local.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         idxs = torch.arange(T, device=x.device)
#         local_mask = (idxs.unsqueeze(0) - idxs.unsqueeze(1)).abs() <= self.local_window
#         local_mask = local_mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
#         scores_local = scores_local.masked_fill(local_mask == 0, float('-inf'))
#         attn_local = self.attn_dropout(F.softmax(scores_local, dim=-1))
#         out_local = torch.matmul(attn_local, V_local)

#         out = torch.cat([out_global, out_local], dim=1)  # (B, num_heads, T, head_dim)
#         out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, d_model)
#         out = self.out_proj(out)
#         return out


# # Multi-Head Attention Block
# class MultiHeadAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads

#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)

#         self.w_o = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(dropout)
#         self.scale = math.sqrt(self.d_k)

#         # Learnable temperature parameter for Gumbel-Softmax (initialized to 1.0)
#         self.log_tau = nn.Parameter(torch.tensor(0.0))  # log(1.0) = 0.0

#     def forward(self, query, key, value, mask=None, stochastic=False):
#         batch_size = query.size(0)

#         Q = self.w_q(query)
#         K = self.w_k(key)
#         V = self.w_v(value)

#         Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

#         scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))

#         if stochastic:
#             tau = torch.exp(self.log_tau)  # ensure tau > 0
#             gumbel_noise = -torch.empty_like(scores).exponential_().log()
#             scores = (scores + gumbel_noise) / tau
#             attn = F.softmax(scores, dim=-1)
#         else:
#             attn = F.softmax(scores, dim=-1)

#         attn = self.dropout(attn)
#         out = torch.matmul(attn, V)

#         out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
#         out = self.w_o(out)

#         return out