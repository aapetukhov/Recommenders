import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=f_size, embedding_dim=embed_dim)
            for f_size in num_features
        ])
        self.embed_dim = embed_dim

    def forward(self, x):
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1)  # (bs, n, d)


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w = nn.Parameter(torch.randn(embed_dim))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.W_Q(x)  # (bs, n, d)
        K = self.W_K(x)  # (bs, n, d)
        V = self.W_V(x)  # (bs, n, d)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.W_Q.out_features ** 0.5)  # (bs, n, n)
        attention_weights = self.softmax(attention_scores)  # (bs, n, n)
        Z = torch.matmul(attention_weights, V)  # (bs, n, d)

        u_weights = self.softmax(torch.matmul(Z, self.w)).unsqueeze(-1)  # (bs, n, 1)
        u = (u_weights * Z).sum(dim=1)  # (bs, d)
        return u


class DeepFM(nn.Module):
    def __init__(self, num_features_kt, num_features_dt, embed_dim):
        super(DeepFM, self).__init__()
        self.kt_feats = EmbeddingLayer(num_features_kt, embed_dim)
        self.dt_feats = EmbeddingLayer(num_features_dt, embed_dim)
        self.kt_embed = AttentionLayer(embed_dim)
        self.dt_embed = AttentionLayer(embed_dim)

    def forward(self, x_kt, x_dt):
        embed_kt = self.kt_feats(x_kt)  # (bs, n, d)
        u_kt = self.kt_embed(embed_kt)  # (bs, d)

        embed_dt = self.dt_feats(x_dt)  # (bs, n, d)
        u_dt = self.dt_embed(embed_dt)  # (bs, d)

        return u_kt, u_dt