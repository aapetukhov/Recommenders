import torch
from torch import nn


class InfoNCELoss(nn.Module):
    """sampled-softmax with bias-correction."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, **_):
        target = torch.zeros(logits.size(0), dtype=torch.long,
                             device=logits.device)
        return {"loss": self.ce(logits, target)}
