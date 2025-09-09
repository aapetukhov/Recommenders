import glob
import gzip
import os
import pickle
import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


class StreamDataset(IterableDataset):
    def __init__(
        self,
        parquet_dir,
        dt_features,
        dt_double_features,
        kt_features,
        kt_double_features,
        dt_emb_path,
        kt_emb_path,
        dt_feat_path,
        kt_feat_path,
        inn_dt_to_idx_path,
        inn_kt_to_idx_path,
        unique_kt_path,
        unique_dt_path,
        label_column="label",
        chunk_size=4096 * 2,
    ):
        self.parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        self.chunk_size = chunk_size

        self.dt_features = dt_features
        self.dt_double_features = dt_double_features
        self.kt_features = kt_features
        self.kt_double_features = kt_double_features
        self.label_column = label_column

        self.dt_emb_path = dt_emb_path

        self.dt_emb = self._load_pickle(dt_emb_path)
        self.kt_emb = self._load_pickle(kt_emb_path)
        self.dt_feat = self._load_pickle(dt_feat_path)
        self.kt_feat = self._load_pickle(kt_feat_path)
        self.unique_kt = self._load_pickle(unique_kt_path)
        self.unique_dt = self._load_pickle(unique_dt_path)

        self.inn_kt_to_index = self._load_pickle(inn_kt_to_idx_path)
        self.inn_dt_to_index = self._load_pickle(inn_dt_to_idx_path)

    def _load_pickle(self, path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    def parse_data(self, chunk):
        for _, row in chunk.iterrows():
            inn_dt, inn_kt = row["inn_dt"], row["inn_kt"]
            if inn_dt not in self.inn_dt_to_index:
                continue
            if inn_kt not in self.inn_kt_to_index:
                continue

            # positive
            yield self.make_sample(inn_dt, inn_kt, label=1)

            # negative
            inn_kt_neg = random.choice(self.unique_kt)
            while inn_kt_neg == inn_kt:
                inn_kt_neg = random.choice(self.unique_kt)
            yield self.make_sample(inn_dt, inn_kt_neg, label=0)

    def make_sample(self, inn_dt, inn_kt, label):
        # positives arrive as raw inn strings
        dt_data = self.dt_feat.get(self.inn_dt_to_index[inn_dt], {})
        kt_data = self.kt_feat.get(self.inn_kt_to_index[inn_kt], {})

        return {
            "label": torch.tensor(label, dtype=torch.float32),
            "dt_emb": torch.tensor(
                self.dt_emb.get(self.inn_dt_to_index[inn_dt], np.zeros(256)),
                dtype=torch.float32,
            ),
            "kt_emb": torch.tensor(
                self.kt_emb.get(self.inn_kt_to_index[inn_kt], np.zeros(256)),
                dtype=torch.float32,
            ),
            "user": torch.tensor(
                [dt_data.get(f, 0) for f in self.dt_features], dtype=torch.long
            ),
            "double_user": torch.tensor(
                [dt_data.get(f, 0.0) for f in self.dt_double_features],
                dtype=torch.float32,
            ),
            "item": torch.tensor(
                [kt_data.get(f, 0) for f in self.kt_features], dtype=torch.long
            ),
            "double_item": torch.tensor(
                [kt_data.get(f, 0.0) for f in self.kt_double_features],
                dtype=torch.float32,
            ),
        }

    def __iter__(self):
        for fpath in self.parquet_files:
            parquet_file = pq.ParquetFile(fpath)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                chunk = batch.to_pandas()
                yield from self.parse_data(chunk)
