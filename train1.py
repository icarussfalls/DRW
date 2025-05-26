import torch
from torch.utils.data import Dataset, DataLoader
from test import build_transformers

# Dummy dataset class
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is not None:
            return x, self.y[idx]
        return x

# Suppose you have your data and targets as numpy arrays
import numpy as np

# Example real data:
X_data = np.random.randn(6000, 128).astype(np.float32)   # Your 6000x128 features
y_data = np.random.randn(6000).astype(np.float32)        # Your targets (regression)

# Convert numpy arrays to torch tensors
X_tensor = torch.from_numpy(X_data)
y_tensor = torch.from_numpy(y_data)

# Create Dataset and DataLoader
dataset = TabularDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Your model (use the class from earlier)
model = build_transformers(num_features=X_data.shape[1], d_model=64, n_heads=8, num_layers=3, d_ff=256, dropout=0.1)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Example training loop snippet
for epoch in range(10):  # example 10 epochs
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = torch.nn.functional.mse_loss(preds, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


