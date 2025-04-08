import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq


class StreamDataset(IterableDataset):
    """
    Streamable Dataset.
    Allows for huuuge data processing without putting it on disk.
    """

    def __init__(
        self,
        file_path: str,
        dt_features: list,
        dt_double_features: list,
        kt_features: list,
        kt_double_features: list,
        label_column="label",
        chunk_size=10000,
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size

        self.dt_features = dt_features # categorical features
        self.dt_double_features = dt_double_features # numeric features
        
        self.kt_features = kt_features
        self.kt_double_features = kt_double_features

        self.label_column = label_column

    def parse_data(self, chunk):
        chunk = chunk.fillna(0)
        
        # user (dt) part - the one who pays
        users = chunk[self.dt_features].values
        users_double = chunk[self.dt_double_features].values.astype(np.float32)

        # item (kt) part - the one who receives the money
        items = chunk[self.kt_features].values
        items_double = chunk[self.kt_double_features].values.astype(np.float32)
        
        labels = chunk[self.label_column].values.astype(np.float32)

        for user, item, double_user, double_item, label in zip(users, items, users_double, items_double, labels):
            yield {
                "user": torch.tensor(
                    user, dtype=torch.long
                ),
                "double_user": torch.tensor(
                    double_user, dtype=torch.float32
                ),
                "item": torch.tensor(
                    item, dtype=torch.long
                ),
                "double_item": torch.tensor(
                    double_item, dtype=torch.float32
                ),
                "label": torch.tensor(label, dtype=torch.float32),  # for BCE loss
            }

    def __iter__(self):
        # maybe pd.read_parquet
        # for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
        #     yield from self.parse_data(chunk)       
        parquet_file = pq.ParquetFile(self.file_path)
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            chunk = batch.to_pandas()
            yield from self.parse_data(chunk)
