import gzip
import os
import pickle
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq


class StreamDataset(IterableDataset):
    def __init__(self, parquet_dir,
                 dt_features, dt_double_features,
                 kt_features, kt_double_features,
                 dt_emb_path, kt_emb_path,
                 dt_feat_path, kt_feat_path,
                 unique_kt_path,
                 neg_weights_path, num_negs,
                 label_column="label", chunk_size=10000):
        self.parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        self.chunk_size = chunk_size

        self.dt_features = dt_features
        self.dt_double_features = dt_double_features
        self.kt_features = kt_features
        self.kt_double_features = kt_double_features
        self.label_column = label_column

        self.dt_emb = self._load_pickle(dt_emb_path)
        self.kt_emb = self._load_pickle(kt_emb_path)
        self.dt_feat = self._load_pickle(dt_feat_path)
        self.kt_feat = self._load_pickle(kt_feat_path)
        self.unique_kt = self._load_pickle(unique_kt_path)

        self.neg_inns, self.neg_weights = self._load_pickle(neg_weights_path)
        self.neg_prob_by_kt = dict(zip(self.neg_inns, self.neg_weights))
        self.num_negs = num_negs

    def _load_pickle(self, path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    def parse_data(self, chunk):
        for _, row in chunk.iterrows():
            inn_dt, inn_kt = row["inn_dt"], row["inn_kt"]
            # positive
            yield self.make_sample(inn_dt, inn_kt)


    def make_sample(self, inn_dt, inn_kt):
        """Возвращает ОДНУ запись: positive + K negative."""
        # positive
        dt_data  = self.dt_feat.get(inn_dt,  {})

        # negatives
        neg_kts  = np.random.choice(self.neg_inns,
                                    size=self.num_negs,
                                    replace=False,
                                    p=self.neg_p)
        kts = np.concatenate(([inn_kt], neg_kts))                 # L = 1+K

        # собираем признаки для каждого item
        cat_items   = torch.tensor([[self.kt_feat.get(k, {}).get(f, 0)
                                     for f in self.kt_features] for k in kts],
                                   dtype=torch.long)              # (L, C_i)
        dbl_items   = torch.tensor([[self.kt_feat.get(k, {}).get(f, 0.0)
                                     for f in self.kt_double_features]
                                     for k in kts],
                                   dtype=torch.float32)           # (L, D_i)
        kt_embs     = torch.tensor([self.kt_emb.get(k, np.zeros(256))
                                    for k in kts], dtype=torch.float32)  # (L,256)

        # вес для bias-коррекции softmax: −log q(k)
        log_q = torch.tensor([0.0] + [-np.log(self.neg_prob_by_kt[k])
                                      for k in neg_kts],
                             dtype=torch.float32)                 # (L,)

        return {
            "user"        : torch.tensor([dt_data.get(f, 0)
                                          for f in self.dt_features], dtype=torch.long),
            "double_user" : torch.tensor([dt_data.get(f, 0.0)
                                          for f in self.dt_double_features], dtype=torch.float32),
            "dt_emb"      : torch.tensor(self.dt_emb.get(inn_dt, np.zeros(256)),
                                         dtype=torch.float32),
            "item"        : cat_items,
            "double_item" : dbl_items,
            "kt_emb"      : kt_embs,
            "log_q"       : log_q,          # для коррекции в loss
        }


    def __iter__(self):
        for fpath in self.parquet_files:
            parquet_file = pq.ParquetFile(fpath)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                chunk = batch.to_pandas()
                yield from self.parse_data(chunk)
