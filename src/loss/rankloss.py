import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.LogSigmoid()

    def forward(self, emb_user, emb_item_pos, emb_item_neg, **batch):
        score_pos = (emb_user * emb_item_pos).sum(dim=1)
        score_new = (emb_user * emb_item_neg).sum(dim=1)

        loss = -self.loss(score_pos - score_new).mean()
        return {"loss": loss}
