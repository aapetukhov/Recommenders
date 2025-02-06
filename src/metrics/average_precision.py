import torch
from torchmetrics import AveragePrecision
from src.metrics.base_metric import BaseMetric


class AP(BaseMetric):
    def __init__(self, name="AP"):
        super().__init__(name)
        self.ap = AveragePrecision(task="binary")
    def __call__(self, **batch):
        return self.ap(batch["logits"], batch["label"]).item()