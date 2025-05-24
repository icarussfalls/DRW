import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Import your model builder (adjust the path accordingly)
#sys.path.append('/kaggle/working/DRW')
from model import build_transformer

# Load data
df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')

# Set features and target
feature_cols = [col for col in df.columns if col != 'label']
target_col = 'label'

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)

X[np.isinf(X)] = 0  # simple fix

# print("Any NaNs in X?", np.isnan(X).any())
# print("Any infs in X?", np.isinf(X).any())
# print("Any NaNs in y?", np.isnan(y).any())
# print("Any infs in y?", np.isinf(y).any())


# Parameters
seq_len = 32       # length of input sequences
val_ratio = 0.1    # 10% for validation
batch_size = 64
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset class
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_len]
        y_seq = self.y[idx + self.seq_len]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32).view(1)

# Create dataset and splits
full_dataset = SequenceDataset(X, y, seq_len)
val_len = int(len(full_dataset) * val_ratio)
train_len = len(full_dataset) - val_len

train_ds, val_ds = torch.utils.data.random_split(
    full_dataset, 
    [train_len, val_len], 
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Model Setup
input_dim = X.shape[1]  # number of features per timestep

model = build_transformer(
    input_dim=input_dim,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    num_layers=6,
    max_seq_len=seq_len,
    output_dim=1,
    dropout=0.1
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create checkpoint directory
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
    for i, (xb, yb) in enumerate(loop):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)  # (batch_size, seq_len, 1)
        preds = preds[:, -1, :]  # last timestep prediction shape: (batch_size, 1)

        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)
        loop.set_postfix(train_loss=train_loss / ((i + 1) * batch_size))

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            preds = preds[:, -1, :]
            val_loss += criterion(preds, yb).item() * xb.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    # Save model checkpoint
    torch.save(model.state_dict(), f"{save_dir}/epoch_{epoch}.pt")
