import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.w = nn.Parameter(torch.randn(embed_dim))
        self.W_u = nn.Linear(embed_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.shape[-1] ** 0.5)
        attention_weights = self.softmax(attention_scores)
        Z = torch.matmul(attention_weights, V)  # [BS, num_feats, embed_dim]
        # TODO: check the math here
        u_weights = self.softmax(self.W_u(Z).squeeze(-1))  # [BS, num_feats]
        u = (u_weights.unsqueeze(-1) * Z).sum(dim=1)

        return self.norm(u)


class FeatureEmbedding(nn.Module):
    """
    Embedding layer. Separate for every feature. When passing features, you need to know
    number of unique values for each feature.

    Keyword arguments:
        feature_sizes : dict -- dict of type {feature_name: num_unique_values}
    """

    def __init__(self, feature_sizes: dict, max_embed_dim=32):
        super(FeatureEmbedding, self).__init__()
        self.embeddings = nn.ModuleDict(
            {
                feat_name: nn.Embedding(
                    num_embeddings=size,
                    embedding_dim=min(int(size**0.5), max_embed_dim),
                )
                for feat_name, size in feature_sizes.items()
            }
        )
        self.feature_dims = {
            feat: min(int(size**0.5), max_embed_dim)
            for feat, size in feature_sizes.items()
        }

    def forward(self, x):
        embedded = [
            self.embeddings[feat_name](x[:, i])
            for i, feat_name in enumerate(self.embeddings)
        ]
        return embedded


class FeatureProjection(nn.Module):
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(in_dim, output_dim) for in_dim in input_dims]
        )

    def forward(self, embedded_features):
        projected = [
            proj(feat) for proj, feat in zip(self.projections, embedded_features)
        ]
        return torch.stack(projected, dim=1)  # (batch_size, num_features, output_dim)


class DeepFM(nn.Module):
    """
    Main model class
    """

    def __init__(self, user_feature_sizes, item_feature_sizes, embed_dim):
        super(DeepFM, self).__init__()
        self.user_embed = FeatureEmbedding(user_feature_sizes, embed_dim)
        self.item_embed = FeatureEmbedding(item_feature_sizes, embed_dim)

        self.user_proj = FeatureProjection(
            list(self.user_embed.feature_dims.values()), embed_dim
        )
        self.item_proj = FeatureProjection(
            list(self.item_embed.feature_dims.values()), embed_dim
        )

        self.user_attention = AttentionLayer(embed_dim)
        self.item_attention = AttentionLayer(embed_dim)

    def forward(self, item, user, **batch):
        user_emb = self.user_proj(self.user_embed(user))  # (batch_size, n, d)
        item_emb = self.item_proj(self.item_embed(item))  # (batch_size, m, d)

        u_kt = self.user_attention(user_emb)  # (batch_size, d)
        u_dt = self.item_attention(item_emb)  # (batch_size, d)

        return {"logits": (u_kt * u_dt).sum(dim=1)}

    def __str__(self):
        """
        Model info with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll params: {all_parameters}"
        result_info = result_info + f"\nTrainable params: {trainable_parameters}"

        return result_info
