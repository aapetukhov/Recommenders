import torch
from src.metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    def __call__(self, **batch):
        logits: torch.Tensor = batch["logits"]
        labels: torch.Tensor = batch["labels"]
        
        predictions = (logits > 0).float()
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()
        
        return accuracy