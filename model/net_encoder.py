import torch
import torch.nn as nn
import math
import numpy as np

from typing import Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, nfeats: int = 256,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 8, num_heads: int = 8,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        
        
        self.layernorm1 = nn.LayerNorm(latent_dim)
        self.sattn_qkv_proj = nn.Linear(latent_dim, latent_dim * 3, bias=False)
        self.sattn_out_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        
        assert latent_dim % num_heads == 0, "d_latent must be divisible by n_heads"
        self.rotary_emb = RotaryEmbedding(dim = latent_dim // num_heads)

        self.layernorm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:

        # 1. Self-Attention (with pre-normalization)
        x_norm = self.layernorm1(x)
        sattn_output = self._sattn(x_norm, attn_mask)
        x = x + sattn_output

        # 2. Feed-Forward Network (with pre-normalization)
        x_norm = self.layernorm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        
        return x

    def _sattn(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        q, k, v = self.sattn_qkv_proj(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b t (nh dh) -> b nh t dh", nh=self.num_heads),
            (q, k, v),
        )

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )

        out = rearrange(out, "b nh t dh -> b t (nh dh)")

        return self.sattn_out_proj(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class MotionEncoder(nn.Module):
    def __init__(self, nfeats: int, vae: bool = True, is_head_encoder=False,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 8, num_heads: int = 8,
                 dropout: float = 0.1, positional_embedding = False,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__()
        self.vae = vae
        input_feats = nfeats
        self.is_head_encoder = is_head_encoder
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))

        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(latent_dim=latent_dim) for _ in range(num_layers)]
        )

        self.output_norm = nn.LayerNorm(latent_dim)

        self.positional_embedding = positional_embedding
        if self.positional_embedding:
            self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout, max_len=300, batch_first=True)

    def forward(self, features: Tensor) -> Union[Tensor, Distribution]:
        B, T, _ = features.shape
        
        if self.is_head_encoder:
            x = features 
        else:
            x = self.skel_embedding(features)  # [B, T, dim]

        if self.vae:
            mu_token = self.mu_token.expand(B, 1, -1) # (B, 1, dim)
            logvar_token = self.logvar_token.expand(B, 1, -1) # (B, 1, dim)
            xseq = torch.cat((mu_token, logvar_token, x), dim=1) # (B, 2+T, dim)
        else:
            xseq = x

        if self.positional_embedding:
            xseq = self.sequence_pos_encoding(xseq)

        for layer in self.encoder_layers:
            xseq = layer(xseq, attn_mask=None)

        final = self.output_norm(xseq)
        
        if self.vae:
            mu, logvar = final[:, 0, :], final[:, 1, :] # (B, 256)           
            std = logvar.exp().pow(0.5)
            return torch.distributions.Normal(mu, std), final[:, 2:, :]
        else:
            return final


