import torch
from torchmetrics import F1Score

from src.metrics.base_metric import BaseMetric


class F1_Score(BaseMetric):
    def __init__(self, name="F1ScoreMetric", threshold=0.0, device="cuda"):
        super().__init__(name)
        self.device = device
        self.f1_score = F1Score(task="binary").to(device)
        self.threshold = threshold

    def __call__(self, **batch):
        preds = (batch["logits"] >= self.threshold).float()
        return self.f1_score(preds, batch["label"]).item()
