import torch
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset


class StreamRetailDataset(IterableDataset):
    """
    Streamable Retail Dataset.
    Allows for huuuge data processing without putting it on disk.
    """
    def __init__(
        self,
        file_path: str,
        dt_features: list,
        kt_features: list,
        label_column="label",
        chunk_size=10000
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.dt_features = dt_features
        self.kt_features = kt_features
        self.label_column = label_column

    def parse_data(self, chunk):
        # TODO: add options for float data to be passed
        chunk = chunk.fillna(0)  # TODO: maybe not 0 ?
        users = chunk[self.dt_features].values
        items = chunk[self.kt_features].values
        labels = chunk[self.label_column].values.astype(np.float32)

        for user, item, label in zip(users, items, labels):
            yield (
                torch.tensor(user, dtype=torch.long), # TODO: fix when there are float features
                torch.tensor(item, dtype=torch.long), # TODO: fix when there are float features
                torch.tensor(label, dtype=torch.float32), # for BCE loss
            )

    def __iter__(self):
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            yield from self.parse_data(chunk)