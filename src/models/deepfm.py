import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, dropout=0.4):
        super(AttentionLayer, self).__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.w = nn.Parameter(torch.randn(embed_dim))
        self.W_u = nn.Linear(embed_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.shape[-1] ** 0.5)
        attention_weights = self.softmax(attention_scores)
        Z = torch.matmul(attention_weights, V)  # [BS, num_feats, embed_dim]
        Z = self.dropout(Z)
        u_weights = self.softmax(self.W_u(Z).squeeze(-1))  # [BS, num_feats]
        u = (u_weights.unsqueeze(-1) * Z).sum(dim=1)

        return self.norm(u), u_weights


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

        embedded = []
        for i, feat_name in enumerate(self.embeddings):
            try:
                embedded.append(self.embeddings[feat_name](x[:, i]))
            except:
                print(f"ERROR WITH {feat_name}")
                raise ValueError("NOOOO...")

        return embedded


class FeatureProjection(nn.Module):
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(in_dim, output_dim) for in_dim in input_dims]
        )
        self.input_dims = input_dims
        self.output_dim = output_dim

    def forward(self, embedded_features):
        projected = []
        for i, (proj, feat) in enumerate(zip(self.projections, embedded_features)):
            try:
                projected.append(proj(feat))
            except:
                print(f"ERROR WITH FEATURE {i}")
                raise ValueError("NOOOO...")
        
        return torch.stack(projected, dim=1)  # (batch_size, num_features, output_dim)


class DeepFM(nn.Module):
    """
    Main model class
    """

    def __init__(
        self,
        num_user_double_feats: int,
        num_item_double_feats: int,
        user_feature_sizes: dict,
        item_feature_sizes: dict,
        embed_dim: int,
    ):
        super(DeepFM, self).__init__()
        self.user_embed = FeatureEmbedding(user_feature_sizes, embed_dim)
        self.item_embed = FeatureEmbedding(item_feature_sizes, embed_dim)

        self.user_proj = FeatureProjection(
            list(self.user_embed.feature_dims.values()), embed_dim
        )
        self.item_proj = FeatureProjection(
            list(self.item_embed.feature_dims.values()), embed_dim
        )

        # branches for continuous features
        self.user_double_proj = nn.Sequential(
            nn.BatchNorm1d(num_user_double_feats, affine=False),
            nn.Linear(num_user_double_feats, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        self.item_double_proj = nn.Sequential(
            nn.BatchNorm1d(num_item_double_feats, affine=False),
            nn.Linear(num_item_double_feats, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        self.user_attention = AttentionLayer(embed_dim)
        self.item_attention = AttentionLayer(embed_dim)

    def forward(self, item, user, double_item, double_user, **batch):
        """
        User = dt
        Item = kt
        """
        u_dt, dt_weights = self.embed_user(user, double_user) # (batch_size, d)
        u_kt, kt_weights = self.embed_item(item, double_item) # (batch_size, d)  

        # TODO: discuss the necessitty of this
        u_dt = F.normalize(u_dt, dim=-1)
        u_kt = F.normalize(u_kt, dim=-1)

        return {
            "logits": (u_kt * u_dt).sum(dim=1),
            "kt_weights": kt_weights,
            "dt_weights": dt_weights,
            }

    def embed_user(self, user, double_user):
        """
        Separate network to embed user (dt).
        """
        user_emb = self.user_proj(self.user_embed(user)) # (batch_size, n, d)
        user_double_projected = self.user_double_proj(double_user).unsqueeze(1) # (batch_size, 1, d)
        user_emb = torch.cat([user_emb, user_double_projected], dim=1) # (batch_size, n + 1, d)
        u_dt, dt_weights = self.user_attention(user_emb) # (batch_size, d)
        return u_dt, dt_weights
    

    def embed_item(self, item, double_item):
        """
        Separate network to embed item (kt).
        """
        item_emb = self.item_proj(self.item_embed(item)) # (batch_size, n, d)
        item_double_projected = self.item_double_proj(double_item).unsqueeze(1) # (batch_size, 1, d)
        item_emb = torch.cat([item_emb, item_double_projected], dim=1) # (batch_size, n + 1, d)
        u_kt, dt_weights = self.item_attention(item_emb) # (batch_size, d)
        return u_kt, dt_weights

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
