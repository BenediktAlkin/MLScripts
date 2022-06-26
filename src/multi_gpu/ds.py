import torch
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.dims = cfg["dims"]
        self.size = cfg["n_batches"] * cfg["batch_size"] - 1

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return torch.full(size=(self.dims,), fill_value=item, dtype=torch.float32)