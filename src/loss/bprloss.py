import torch.nn as nn
import torch.functional as F


class BPRLoss(nn.Module):
    def forward(self, embedding_user, embedding_item_pos, embedding_item_neg, **batch):
        score_pos = (embedding_user * embedding_item_pos).sum(dim=1)
        score_neg = (embedding_user * embedding_item_neg).sum(dim=1)
        loss = -F.logsigmoid(score_pos - score_neg).mean()
        return {"loss": loss}