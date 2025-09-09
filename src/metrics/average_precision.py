import torch
from torchmetrics import AveragePrecision

from src.metrics.base_metric import BaseMetric


class AUC_PR(BaseMetric):
    def __init__(self, name="AP", device="cuda"):
        super().__init__(name)
        self.device = device
        self.aupr = AveragePrecision(task="binary").to(device)

    def __call__(self, **batch):
        return self.aupr(batch["logits"], batch["label"].long()).item()
