import torch
import torch.nn as nn

from src.models.deepfm import AttentionLayer, FeatureEmbedding, FeatureProjection


class DeepFMReverse(nn.Module):
    """
    DeepFM variant for kt->dt direction.

    User tower operates on kt descriptors while item tower embeds dt passports.
    """

    def __init__(
        self,
        num_user_double_feats: int,
        num_item_double_feats: int,
        user_feature_sizes: dict,
        item_feature_sizes: dict,
        embed_dim: int,
        num_user_topic_embeds: int = 0,
        num_item_topic_embeds: int = 0,
    ):
        super().__init__()
        self.user_embed = FeatureEmbedding(user_feature_sizes, embed_dim)
        self.item_embed = FeatureEmbedding(item_feature_sizes, embed_dim)

        self.user_proj = FeatureProjection(
            list(self.user_embed.feature_dims.values()), embed_dim
        )
        self.item_proj = FeatureProjection(
            list(self.item_embed.feature_dims.values()), embed_dim
        )

        self.user_double_proj = nn.Sequential(
            nn.BatchNorm1d(num_user_double_feats, affine=False),
            nn.Linear(num_user_double_feats, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.item_double_proj = nn.Sequential(
            nn.BatchNorm1d(num_item_double_feats, affine=False),
            nn.Linear(num_item_double_feats, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        self.user_attention = AttentionLayer(embed_dim)
        self.item_attention = AttentionLayer(embed_dim)

        self.kt_emb_proj = nn.Sequential(
            nn.BatchNorm1d(256, affine=False),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.dt_emb_proj = nn.Sequential(
            nn.BatchNorm1d(256, affine=False),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        self.kt_topic_emb_proj = (
            nn.Sequential(
                nn.BatchNorm1d(num_user_topic_embeds, affine=False),
                nn.Linear(num_user_topic_embeds, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
            )
            if num_user_topic_embeds > 0
            else None
        )
        self.dt_topic_emb_proj = (
            nn.Sequential(
                nn.BatchNorm1d(num_item_topic_embeds, affine=False),
                nn.Linear(num_item_topic_embeds, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
            )
            if num_item_topic_embeds > 0
            else None
        )

    def forward(
        self,
        item,
        user,
        double_item,
        double_user,
        dt_emb=None,
        kt_emb=None,
        dt_topic_emb=None,
        kt_topic_emb=None,
        **batch,
    ):
        """
        User = kt
        Item = dt
        """
        u_kt, kt_weights = self.embed_user(user, double_user, kt_emb, kt_topic_emb)
        u_dt, dt_weights = self.embed_item(item, double_item, dt_emb, dt_topic_emb)

        return {
            "logits": (u_kt * u_dt).sum(dim=1),
            "kt_weights": kt_weights,
            "dt_weights": dt_weights,
        }

    def embed_user(self, user, double_user, kt_emb=None, kt_topic_emb=None):
        """
        Separate network to embed user (kt).
        """
        user_emb = self.user_proj(self.user_embed(user))
        user_double = self.user_double_proj(double_user).unsqueeze(1)
        parts = [user_emb, user_double]

        if kt_emb is not None:
            projected = self.kt_emb_proj(kt_emb).unsqueeze(1)
            parts.append(projected)

        if kt_topic_emb is not None and self.kt_topic_emb_proj is not None:
            projected = self.kt_topic_emb_proj(kt_topic_emb).unsqueeze(1)
            parts.append(projected)

        x = torch.cat(parts, dim=1)
        u_kt, weights = self.user_attention(x)
        return u_kt, weights

    def embed_item(self, item, double_item, dt_emb=None, dt_topic_emb=None):
        """
        Separate network to embed item (dt).
        """
        item_emb = self.item_proj(self.item_embed(item))
        item_double = self.item_double_proj(double_item).unsqueeze(1)
        parts = [item_emb, item_double]

        if dt_emb is not None:
            projected = self.dt_emb_proj(dt_emb).unsqueeze(1)
            parts.append(projected)

        if dt_topic_emb is not None and self.dt_topic_emb_proj is not None:
            projected = self.dt_topic_emb_proj(dt_topic_emb).unsqueeze(1)
            parts.append(projected)

        x = torch.cat(parts, dim=1)
        u_dt, weights = self.item_attention(x)
        return u_dt, weights

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll params: {all_parameters}"
        result_info = result_info + f"\nTrainable params: {trainable_parameters}"
        return result_info
