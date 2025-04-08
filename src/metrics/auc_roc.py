import torch
from torchmetrics import AUROC
# from sklearn.metrics import roc_auc_score

from src.metrics.base_metric import BaseMetric


class AUC_ROC(BaseMetric):
    def __init__(self, name="AUC_ROC"):
        super().__init__(name)
        self.auroc = AUROC(task="binary")
    def __call__(self, **batch):
        return self.auroc(batch["logits"], batch["label"]).item()
