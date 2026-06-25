import torch
import torch.nn as nn
import numpy as np


from typing import Optional, Union, Tuple, Dict
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from model.net_encoder_re import MotionEncoder
from model.net_decoder_re import MotionDecoder
from quat import *

class Head2Motion(nn.Module):
    """ Head to Motion 파이프라인 """
    def __init__(self, head_dim: int, motion_dim: int, out_dim: int, vae: bool = True,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 # --- NEW: masking controls ---
                 mask_prob: float = 0.5,          # 배치에서 마스킹을 켤 확률
                 min_valid_len: int = 8,          # L의 최소값
                 last_frame_weight: float = 0.0,  # loss에서 마지막 프레임 가중치(원하면 loss쪽에서 사용)
                 **kwargs
        ):
        super().__init__()
        self.vae = vae
        self.seq_len = 128
        
        # --- NEW ---
        self.mask_prob = float(mask_prob)
        self.min_valid_len = int(min_valid_len)
        self.last_frame_weight = float(last_frame_weight)

        self.head_tokenizer = MotionEncoder(nfeats=head_dim, vae=False, latent_dim=latent_dim,
                                          ff_size=ff_size, num_layers=num_layers,
                                          num_heads=num_heads, dropout=dropout,
                                          activation=activation)
        
        self.learnable_tokens = nn.Parameter(torch.randn(self.seq_len, latent_dim))
        self.concat_mixer = nn.Linear(latent_dim*2, latent_dim)

        self.pose_embedding = nn.Sequential(
                              nn.Linear(126, 512),
                              nn.LeakyReLU(0.1),
                              nn.Linear(512, latent_dim),
                          )
        
        self.head_encoder = MotionEncoder(nfeats=head_dim, vae=vae, latent_dim=latent_dim, is_head_encoder=True,
                                          ff_size=ff_size, num_layers=num_layers,
                                          num_heads=num_heads, dropout=dropout, positional_embedding=False,
                                          activation=activation)
        
        self.motion_encoder = MotionEncoder(nfeats=motion_dim, vae=vae, latent_dim=latent_dim, is_head_encoder=True,
                                            ff_size=ff_size, num_layers=num_layers,
                                            num_heads=num_heads, dropout=dropout, positional_embedding=False,
                                            activation=activation)
        
        self.motion_decoder = MotionDecoder(nfeats=out_dim, latent_dim=latent_dim,
                                            ff_size=ff_size, num_layers=num_layers,
                                            num_heads=num_heads, dropout=dropout,
                                            activation=activation)
        
        self.head_feat_dropout = nn.Dropout(p=dropout)
        self.motion_feat_dropout = nn.Dropout(p=dropout)
        
        self.head_betas_layer = nn.Linear(latent_dim, 16)
        self.head_contacts_layer = nn.Linear(latent_dim, 21)
        self.head_theta_layer = nn.Linear(latent_dim, 126)

        self.head_mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

    def _sample_valid_lens(self, B: int, T: int, device) -> Tensor:
        # L ~ [min_valid_len, T]
        low = min(self.min_valid_len, T)
        return torch.randint(low=low, high=T + 1, size=(B,), device=device)

    def _build_time_mask(self, valid_lens: Tensor, T: int) -> Tensor:
        # time_mask[b,t] = True if t < L_b else False
        t = torch.arange(T, device=valid_lens.device)[None, :]  # (1,T)
        return t < valid_lens[:, None]  # (B,T) bool

    def forward(self, head: Tensor, motion: Tensor) -> Dict[str, Union[Tensor, Distribution]]:
        B, T, _ = head.shape
        assert T == self.seq_len, f"train forward expects T={self.seq_len}, got T={T}"
        # motion = 6 + 126

        motion = motion[..., 6:132]
        motion_tokens = self.pose_embedding(motion)

        head_tokens = self.head_tokenizer(head)

        # --- NEW: apply prefix-length (tail) mask sometimes ---
        do_mask = self.training and (self.mask_prob > 0.0) and (torch.rand((), device=head.device) < self.mask_prob)

        if do_mask:
            valid_lens = self._sample_valid_lens(B, T, head.device)    # (B,)
            time_mask = self._build_time_mask(valid_lens, T)           # (B,T) True=valid
            # invalid positions:
            inv = ~time_mask                                            # (B,T)
            # replace invalid head tokens with learnable mask token
            head_tokens = head_tokens.clone()
            head_tokens[inv] = self.head_mask_token.expand(B, T, -1)[inv]
        else:
            valid_lens = torch.full((B,), T, device=head.device, dtype=torch.long)
            time_mask = torch.ones((B, T), device=head.device, dtype=torch.bool)


        head_tokens = torch.cat((head_tokens, self.learnable_tokens.expand(B, -1, -1)), dim=-1)
        head_tokens = self.concat_mixer(head_tokens) # (B, T, dim)
        
        dist_head, feat_head = self.head_encoder(head_tokens) 
        dist_motion, feat_motion = self.motion_encoder(motion_tokens)

        if self.vae:
            z_head = dist_head.rsample()
            z_motion = dist_motion.rsample()

        feat_head_dropped = self.head_feat_dropout(feat_head)
        feat_motion_dropped = self.motion_feat_dropout(feat_motion)
        
        head_out = self.motion_decoder(feat_head_dropped, z_head) # (B, T, 256)
        motion_out = self.motion_decoder(feat_motion_dropped, z_motion) # (B, T, 256)
        
        head_betas = self.head_betas_layer(head_out)
        head_contacts = self.head_contacts_layer(head_out)
        head_theta = self.head_theta_layer(head_out)
        
        motion_betas = self.head_betas_layer(motion_out)
        motion_contacts = self.head_contacts_layer(motion_out)
        motion_theta = self.head_theta_layer(motion_out)
        
        return {
            "head_betas": head_betas,           # (B, T, 16)
            "head_contacts": head_contacts,     # (B, T, 21)
            "head_theta": head_theta,           # (B, T, 126)
            "head_dist": dist_head,
            "head_z": z_head,                   # (B, 256)
            
            
            "motion_betas": motion_betas,       # (B, T, 16)
            "motion_contacts": motion_contacts, # (B, T, 21)
            "motion_theta": motion_theta,       # (B, T, 126)
            "motion_dist": dist_motion,
            "motion_z": z_motion,               # (B, 256)
    
            # --- NEW ---
            "time_mask": time_mask,      # (B,T) True=valid
            "valid_lens": valid_lens,    # (B,)

        }

    def forward_test(self, head: Tensor, motion: Tensor) -> Dict[str, Union[Tensor, Distribution]]:
        B, T, _ = head.shape
        slice_token = self.learnable_tokens[:T,...]

        motion = motion[..., 6:132]
        motion_tokens = self.pose_embedding(motion)

        head_tokens = self.head_tokenizer(head)
        head_tokens = torch.cat((head_tokens, slice_token.expand(B, -1, -1)), dim=-1)
        head_tokens = self.concat_mixer(head_tokens) # (B, T, dim)
        
        dist_head, feat_head = self.head_encoder(head_tokens) 
        dist_motion, feat_motion = self.motion_encoder(motion_tokens)

        if self.vae:
            z_head = dist_head.rsample()
            z_motion = dist_motion.rsample()
            
        feat_head_dropped = self.head_feat_dropout(feat_head)
        feat_motion_dropped = self.motion_feat_dropout(feat_motion)
        
        head_out = self.motion_decoder(feat_head_dropped, z_head) # (B, T, 256)
        motion_out = self.motion_decoder(feat_motion_dropped, z_motion) # (B, T, 256)
        
        head_betas = self.head_betas_layer(head_out)
        # head_betas = torch.zeros(B, T, 16, device=head.device)
        head_contacts = self.head_contacts_layer(head_out)
        head_theta = self.head_theta_layer(head_out)
        
        # motion_betas = self.motion_betas_layer(motion_out)
        # motion_contacts = self.motion_contacts_layer(motion_out)
        # motion_theta = self.motion_theta_layer(motion_out)
        
        pose_quat = matrix_to_quaternion(rotation_6d_to_matrix(head_theta.reshape(B, T, -1, 6)))
        return {
            "head_betas": head_betas,           # (B, T, 16)
            "head_contacts": head_contacts,     # (B, T, 21)
            "head_theta": head_theta,           # (B, T, 126) 
            "head_dist": dist_head,
            "head_z": z_head,                   # (B, 256)            
            "pose_quat": pose_quat,             # (B, T, 21, 4)  
            # "motion_betas": motion_betas,       # (B, T, 16)
            # "motion_contacts": motion_contacts, # (B, T, 21)
            # "motion_theta": motion_theta,       # (B, T, 126)
            "motion_dist": dist_motion,
            "motion_z": z_motion,               # (B, 256)
        }