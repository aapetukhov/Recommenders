import torch
from src.metrics.base_metric import BaseMetric


class HitAtK(BaseMetric):
    def __init__(self, k: int, name=None):
        super().__init__(name or f"Hit@{k}")
        self.k = k

    def __call__(self, *, logits: torch.Tensor, **_):
        hit = (logits.topk(self.k, dim=1).indices == 0).any(dim=1).float()
        return hit.mean().item()


class MRR(BaseMetric):
    def __init__(self, name="MRR"):
        super().__init__(name)

    def __call__(self, *, logits: torch.Tensor, **_):
        ranks = (logits.argsort(dim=1, descending=True) == 0).nonzero()[:, 1]
        return (1.0 / (ranks + 1).float()).mean().item()


class NDCGAtK(BaseMetric):
    def __init__(self, k: int, name=None):
        super().__init__(name or f"NDCG@{k}")
        self.k = k
        self.log2 = torch.log2

    def __call__(self, *, logits: torch.Tensor, **_):
        order = logits.argsort(dim=1, descending=True)
        ranks = (order == 0).nonzero()[:, 1]
        in_topk = (ranks < self.k).float()
        dcg = in_topk / self.log2(ranks.float() + 2)
        return dcg.mean().item()
