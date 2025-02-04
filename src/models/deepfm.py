import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w = nn.Parameter(torch.randn(embed_dim))
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.shape[-1] ** 0.5)
        attention_weights = self.softmax(attention_scores)
        Z = torch.matmul(attention_weights, V)

        u_weights = self.softmax(torch.matmul(Z, self.w)).unsqueeze(-1)
        u = (u_weights * Z).sum(dim=1)

        return self.norm(u + torch.mean(x, dim=1))


class FeatureEmbedding(nn.Module):
    """Embedding layer. Separate for every feature. When passing features, you need to know
    number of unique values for each feature.

    Keyword arguments:
        feature_sizes : dict -- dict of type {feature_name: num_unique_values}
    """
    def __init__(self, feature_sizes: dict, embed_dim: int):
        super(FeatureEmbedding, self).__init__()
        self.embeddings = nn.ModuleDict({
            feat_name: nn.Embedding(num_embeddings=size, embedding_dim=embed_dim)
            for feat_name, size in feature_sizes.items()
        })
        self.feature_names = list(feature_sizes.keys())

    def forward(self, x):
        embedded = [self.embeddings[feat_name](x[:, i]) for i, feat_name in enumerate(self.feature_names)]
        return torch.stack(embedded, dim=1)  # (batch_size, num_features, embed_dim)


class DeepFM(nn.Module):
    """
    Main model class
    """
    
    def __init__(self, user_feature_sizes, item_feature_sizes, embed_dim):
        super(DeepFM, self).__init__()
        self.user_embed = FeatureEmbedding(user_feature_sizes, embed_dim)
        self.item_embed = FeatureEmbedding(item_feature_sizes, embed_dim)
        self.user_attention = AttentionLayer(embed_dim)
        self.item_attention = AttentionLayer(embed_dim)

    def forward(self, user, item):
        user_emb = self.user_embed(user)  # (batch_size, n, d)
        item_emb = self.item_embed(item)  # (batch_size, m, d)

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
