import torch
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset


class RetailDataset(IterableDataset):
    def __init__(self, file_path, dt_features, kt_features, chunk_size=10000, **kwargs):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.dt_features = dt_features
        self.kt_features = kt_features

    def parse_data(self, chunk):
        users = chunk[self.dt_features].values
        items = chunk[self.kt_features].values
        labels = chunk['label'].values.astype(np.float32)

        for user, item, label in zip(users, items, labels):
            yield torch.tensor(user, dtype=torch.long), \
                  torch.tensor(item, dtype=torch.long), \
                  torch.tensor(label, dtype=torch.float32)

    def __iter__(self):
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            yield from self.parse_data(chunk)