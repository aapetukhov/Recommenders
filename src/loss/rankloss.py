import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.LogSigmoid()

    def forward(self, embedding_user, embedding_item_pos, embedding_item_neg, **batch):
        score_pos = (embedding_user * embedding_item_pos).sum(dim=1)
        score_neg = (embedding_user * embedding_item_neg).sum(dim=1)
        
        loss = - self.loss(score_pos - score_neg).mean()
        return {"loss": loss}
