from torch.utils.data import Dataset, DataLoader


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