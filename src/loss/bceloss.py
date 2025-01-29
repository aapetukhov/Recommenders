import torch
from torch import nn


class BCELoss(nn.Module):
    """
    BCE loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        return {"loss": self.loss(logits, labels)}
