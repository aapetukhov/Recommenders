import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2,
        num_heads: int = 1,
        kt_features: dict = None,
        dt_features: dict = None,
    ):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), f"embedding_dim must be divisible by num_heads,\ngot embed dim={embedding_dim} and num_heads={num_heads}"

        self.embeddings_kt = nn.ModuleDict(
            {
                feat: nn.Embedding(num_categories, embedding_dim)
                for feat, num_categories in kt_features.items()
            }
        )
        self.embeddings_dt = nn.ModuleDict(
            {
                feat: nn.Embedding(num_categories, embedding_dim)
                for feat, num_categories in dt_features.items()
            }
        )

        # separate networks for kt and dt
        self.attention_kt = nn.MultiheadAttention(embedding_dim, num_heads)
        self.attention_dt = nn.MultiheadAttention(embedding_dim, num_heads)

        self.weights_kt = nn.Linear(in_features=embedding_dim, out_features=1)
        self.weights_dt = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(
        self,
        kt_features: torch.Tensor,
        dt_features: torch.Tensor,
        **batch,
    ) -> torch.Tensor:
        kt_embeds = torch.stack(
            [self.embeddings_kt[feat](kt_features[feat]) for feat in kt_features], dim=1
        )
        dt_embeds = torch.stack(
            [self.embeddings_dt[feat](dt_features[feat]) for feat in dt_features], dim=1
        )

        kt_attn_output, _ = self.attention_kt(kt_embeds, kt_embeds, kt_embeds)
        dt_attn_output, _ = self.attention_dt(dt_embeds, dt_embeds, dt_embeds)

        kt_pooled = (
            torch.softmax(self.weights_kt(kt_attn_output).squeeze(-1), dim=-1)
            @ kt_attn_output
        )
        dt_pooled = (
            torch.softmax(self.weights_dt(dt_attn_output).squeeze(-1), dim=-1)
            @ dt_attn_output
        )

        e_user = kt_pooled
        e_item = dt_pooled
        return e_user, e_item
