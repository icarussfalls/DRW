import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import os
from model import build_transformer

# ========= Dataset Definition =========
class ParquetTabularDataset(Dataset):
    def __init__(self, file_path, target_column="target"):
        self.table = pq.read_table(file_path)
        self.df = self.table.to_pandas()  # You could use chunks for >10M rows
        self.target_column = target_column

        self.features = self.df.drop(columns=[target_column]).values.astype(np.float32)
        self.targets = self.df[target_column].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.from_numpy(self.targets[idx])
        return x, y

# ========= Load Dataset =========
file_path = "train.parquet"
target_column = "target"
dataset = ParquetTabularDataset(file_path, target_column)

# ========= Train/Val Split =========
val_ratio = 0.1
total_len = len(dataset)
val_len = int(val_ratio * total_len)
train_len = total_len - val_len

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# ========= Model Setup =========

model = model = build_transformer(input_dim=800, d_model=512, num_heads=8, d_ff=2048,
                              num_layers=6, max_seq_len=100, output_dim=1, dropout=0.1).to("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========= Training Loop =========
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1, 11):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(model.device), yb.to(model.device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(model.device), yb.to(model.device)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
    torch.save(model.state_dict(), f"{save_dir}/epoch_{epoch}.pt")
