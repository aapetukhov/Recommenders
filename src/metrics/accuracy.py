import torch

from src.metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    def __call__(self, **batch):
        logits: torch.Tensor = batch["logits"]
        label: torch.Tensor = batch["label"]

        predictions = (logits > 0).float()
        correct = (predictions == label).float()
        accuracy = correct.mean().item()

        return accuracy
