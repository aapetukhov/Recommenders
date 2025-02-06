import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.metrics.base_metric import BaseMetric


class AUC_ROC(BaseMetric):
    """
    Compute Area Under the Curve (AUC) from batch logits and labels.
    """

    def __init__(self, name="AUC_ROC"):
        super().__init__(name)

    def __call__(self, **batch):
        logits = batch["logits"].detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        return roc_auc_score(labels, logits)
