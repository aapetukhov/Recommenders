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
                 inn_dt_to_idx_path, inn_kt_to_idx_path, 
                 unique_kt_path, unique_dt_path,
                 label_column="label", chunk_size=4096*2):
        
        print("INITIALIZING DATASET")

        self.parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        self.chunk_size = chunk_size

        self.dt_features = dt_features
        self.dt_double_features = dt_double_features
        self.kt_features = kt_features
        self.kt_double_features = kt_double_features
        self.label_column = label_column
        
        self.dt_emb_path = dt_emb_path

        print("LOADING DATA")

        self.dt_emb = self._load_pickle(dt_emb_path)
        self.kt_emb = self._load_pickle(kt_emb_path)
        self.dt_feat = self._load_pickle(dt_feat_path)
        self.kt_feat = self._load_pickle(kt_feat_path)
        self.unique_kt = self._load_pickle(unique_kt_path)
        self.unique_dt = self._load_pickle(unique_dt_path)
        
        print("LOADING INN TO INDEX MAPPINGS")

        self.inn_kt_to_index = self._load_pickle(inn_kt_to_idx_path)
        self.inn_dt_to_index = self._load_pickle(inn_dt_to_idx_path)
        
        print("INITIALIZED DATASET")

    def _load_pickle(self, path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    def parse_data(self, chunk):
        for _, row in chunk.iterrows():
            inn_dt, inn_kt_pos = row["inn_dt"], row["inn_kt"]
            inn_kt_neg = random.choices(self.unique_kt, k=1)[0]
            if inn_dt not in self.inn_dt_to_index:
                continue
            if inn_kt_pos not in self.inn_kt_to_index:
                continue
            if inn_kt_pos == inn_kt_neg:
                continue
            
            yield self.make_bpr_sample(inn_dt, inn_kt_pos, inn_kt_neg)

    def make_sample(self, inn_dt, inn_kt, label):
        # positives arrive as raw inn strings
        dt_data = self.dt_feat.get(self.inn_dt_to_index[inn_dt], {})
        kt_data = self.kt_feat.get(self.inn_kt_to_index[inn_kt], {})

        return {
            "label": torch.tensor(label, dtype=torch.float32),
            "dt_emb": torch.tensor(self.dt_emb.get(self.inn_dt_to_index[inn_dt], np.zeros(256)), dtype=torch.float32),
            "kt_emb": torch.tensor(self.kt_emb.get(self.inn_kt_to_index[inn_kt], np.zeros(256)), dtype=torch.float32),
            "user": torch.tensor([dt_data.get(f, 0) for f in self.dt_features], dtype=torch.long),
            "double_user": torch.tensor([dt_data.get(f, 0.0) for f in self.dt_double_features], dtype=torch.float32),
            "item": torch.tensor([kt_data.get(f, 0) for f in self.kt_features], dtype=torch.long),
            "double_item": torch.tensor([kt_data.get(f, 0.0) for f in self.kt_double_features], dtype=torch.float32),
        }

  
    def make_bpr_sample(self, inn_dt, inn_kt_pos, inn_kt_neg):
        # positives arrive as raw inn strings
        dt_idx = self.inn_dt_to_index[inn_dt]
        pos_idx = self.inn_kt_to_index[inn_kt_pos]
        neg_idx = self.inn_kt_to_index[inn_kt_neg]

        dt_data = self.dt_feat.get(dt_idx, {})
        kt_data_pos = self.kt_feat.get(pos_idx, {})
        kt_data_neg = self.kt_feat.get(self.inn_kt_to_index[inn_kt_neg], {})

        return {
            #user info
            "dt_emb": torch.tensor(self.dt_emb.get(dt_idx, np.zeros(256)), dtype=torch.float32),
            "user": torch.tensor([dt_data.get(f, 0) for f in self.dt_features], dtype=torch.long),
            "double_user": torch.tensor([dt_data.get(f, 0.0) for f in self.dt_double_features], dtype=torch.float32),

            #positive item info
            "kt_emb_pos": torch.tensor(self.kt_emb.get(pos_idx, np.zeros(256)), dtype=torch.float32),
            "item_pos": torch.tensor([kt_data_pos.get(f, 0) for f in self.kt_features], dtype=torch.long),
            "double_item_pos": torch.tensor([kt_data_pos.get(f, 0.0) for f in self.kt_double_features], dtype=torch.float32),
            
            # negative sample info
            "kt_emb_neg": torch.tensor(self.kt_emb.get(neg_idx, np.zeros(256)), dtype=torch.float32),
            "item_neg": torch.tensor([kt_data_neg.get(f, 0) for f in self.kt_features], dtype=torch.long),
            "double_item_neg": torch.tensor([kt_data_neg.get(f, 0.0) for f in self.kt_double_features], dtype=torch.float32),
        }
        

    def __iter__(self):
        for fpath in self.parquet_files:
            parquet_file = pq.ParquetFile(fpath)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                chunk = batch.to_pandas()
                yield from self.parse_data(chunk)
