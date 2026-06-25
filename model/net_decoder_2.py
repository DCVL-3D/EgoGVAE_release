import torch
import torch.nn as nn

from typing import Optional, Union
from torch import nn, Tensor

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, nfeats: int,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 8, num_heads: int = 8,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        
        # 1. Masked Self-Attention Layers
        self.layernorm1 = nn.LayerNorm(latent_dim)
        self.sattn_qkv_proj = nn.Linear(latent_dim, latent_dim * 3, bias=False)
        self.sattn_out_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.rotary_emb = RotaryEmbedding(dim = latent_dim // num_heads)

        # 2. Cross-Attention Layers
        self.layernorm2 = nn.LayerNorm(latent_dim)
        # Query는 디코더의 입력(x)에서, Key와 Value는 인코더 출력에서 가져옵니다.
        self.cattn_q_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.cattn_kv_proj = nn.Linear(latent_dim, latent_dim * 2, bias=False)
        self.cattn_out_proj = nn.Linear(latent_dim, latent_dim, bias=False)

        # 3. Feed-Forward Network Layers
        self.layernorm3 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
    ) -> Tensor:

        # 1. Masked Self-Attention (with pre-normalization)
        q_norm = self.layernorm1(q)
        sattn_output = self._sattn(q_norm, self_attn_mask)
        q = q + sattn_output

        # 2. Cross-Attention (with pre-normalization)
        q_norm = self.layernorm2(q)
        cattn_output = self._cattn(q_norm, kv, cross_attn_mask)
        q = q + cattn_output
        
        # 3. Feed-Forward Network (with pre-normalization)
        q_norm = self.layernorm3(q)
        mlp_output = self.mlp(q_norm)
        q = q + mlp_output
        
        return q

    def _sattn(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        # Q, K, V 프로젝션
        q, k, v = self.sattn_qkv_proj(x).chunk(3, dim=-1)

        # Multi-head 형태로 재배열
        q, k, v = map(
            lambda t: rearrange(t, "b t (nh dh) -> b nh t dh", nh=self.num_heads),
            (q, k, v),
        )

        # Rotary Positional Embedding 적용
        # q = self.rotary_emb.rotate_queries_or_keys(q)
        # k = self.rotary_emb.rotate_queries_or_keys(k)

        # Scaled Dot-Product Attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )

        out = rearrange(out, "b nh t dh -> b t (nh dh)")
        return self.sattn_out_proj(out)

    def _cattn(self, x: Tensor, z: Tensor, attn_mask: Tensor | None) -> Tensor:
        # Query는 디코더의 현재 상태(x)에서 프로젝션
        q = self.cattn_q_proj(x)
        
        # Key와 Value는 인코더의 최종 출력(encoder_output)에서 프로젝션
        k, v = self.cattn_kv_proj(z).chunk(2, dim=-1)

        # Multi-head 형태로 재배열
        q, k, v = map(
            lambda t: rearrange(t, "b t (nh dh) -> b nh t dh", nh=self.num_heads),
            (q, k, v),
        )
        
        # Scaled Dot-Product Attention (Cross-Attention에는 보통 Rotary Embedding을 적용하지 않음)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )

        out = rearrange(out, "b nh t dh -> b t (nh dh)")
        return self.cattn_out_proj(out)


class MotionDecoder(nn.Module):
    def __init__(self, nfeats: int,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 6, num_heads: int = 6,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__()

        output_feats = nfeats
        
        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(nfeats=nfeats, latent_dim=latent_dim) for _ in range(num_layers)]
        )

        self.output_norm = nn.LayerNorm(latent_dim)

        # self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, feat: Tensor, z: Tensor) -> Tensor:
        # z: (B, 256)
        # feat: (B, T, 256)
        
        B, T, _ = feat.shape
        
        z = z.unsqueeze(1)
        
        for layer in self.decoder_layers:
            feat = layer(feat, z)

        out = self.output_norm(feat)
        
        # final = self.final_layer(out) # (B, T, 132)
        # return final
        return out