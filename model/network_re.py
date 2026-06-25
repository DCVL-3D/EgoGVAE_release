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
                 activation: str = "gelu", **kwargs):
        super().__init__()
        self.vae = vae
        self.seq_len = 128
        
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
        
    def forward(self, head: Tensor, motion: Tensor) -> Dict[str, Union[Tensor, Distribution]]:
        B, T, _ = head.shape
        # motion = 6 + 126

        motion = motion[..., 6:132]
        motion_tokens = self.pose_embedding(motion)

        head_tokens = self.head_tokenizer(head)
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
        }

    def forward_test(self, head: Tensor, motion: Tensor) -> Dict[str, Union[Tensor, Distribution]]:
        B, T, _ = head.shape
        print(head.shape)
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
    
    def forward_realworld(self, head: Tensor, motion: Tensor) -> Dict[str, Union[Tensor, Distribution]]:
        from torch.distributions import Normal
        B, T, D = head.shape
        device = head.device

        slice_token = self.learnable_tokens[:T, ...]

        head_tokens = self.head_tokenizer(head)
        head_tokens = torch.cat((head_tokens, slice_token.expand(B, -1, -1)), dim=-1)
        head_tokens = self.concat_mixer(head_tokens) # (B, T, dim)

        dist_head, feat_head = self.head_encoder(head_tokens) 
        z_head = dist_head.rsample()

        feat_head_dropped = self.head_feat_dropout(feat_head)
        head_out = self.motion_decoder(feat_head_dropped, z_head) 

        head_betas = self.head_betas_layer(head_out)
        head_contacts = self.head_contacts_layer(head_out)
        head_theta = self.head_theta_layer(head_out)

        pose_quat = matrix_to_quaternion(rotation_6d_to_matrix(head_theta.reshape(B, T, -1, 6)))

        latent_dim = z_head.shape[-1] 
        dummy_mean = torch.zeros((B, latent_dim), device=device)
        dummy_std = torch.ones((B, latent_dim), device=device)
        dist_motion = Normal(dummy_mean, dummy_std)
        z_motion = torch.zeros((B, latent_dim), device=device)

        return {
            "head_betas": head_betas,           # (B, T, 16)
            "head_contacts": head_contacts,     # (B, T, 21)
            "head_theta": head_theta,           # (B, T, 126) 
            "head_dist": dist_head,
            "head_z": z_head,                   # (B, 256)            
            "pose_quat": pose_quat,             # (B, T, 21, 4)  
            "motion_dist": dist_motion,         # Dummy distribution
            "motion_z": z_motion,               # Dummy latent (B, 256)
            "feat_head": feat_head,
        }

    def forward_sampling(
        self,
        head: Tensor,
        motion: Tensor,
        num_samples: int = 1,
    ) -> Dict[str, Union[Tensor, Distribution]]:
        B, T, _ = head.shape
        slice_token = self.learnable_tokens[:T, ...]

        motion = motion[..., 6:132]
        motion_tokens = self.pose_embedding(motion)

        head_tokens = self.head_tokenizer(head)
        head_tokens = torch.cat((head_tokens, slice_token.expand(B, -1, -1)), dim=-1)
        head_tokens = self.concat_mixer(head_tokens)  # (B, T, dim)

        dist_head, feat_head = self.head_encoder(head_tokens)
        dist_motion, feat_motion = self.motion_encoder(motion_tokens)

        if self.vae:
            # z_head: (num_samples, B, dim)
            z_head = dist_head.rsample((num_samples,))
            z_motion = dist_motion.rsample()
        else:
            raise NotImplementedError("num_samples requires VAE sampling.")

        feat_head_dropped = self.head_feat_dropout(feat_head)

        # ---------------------------------------------------------
        # Expand feature sequence for multiple z samples.
        # feat_head_dropped: (B, T, dim)
        # -> (num_samples, B, T, dim)
        # -> (num_samples * B, T, dim)
        # ---------------------------------------------------------
        feat_head_rep = (
            feat_head_dropped.unsqueeze(0)
            .expand(num_samples, -1, -1, -1)
            .reshape(num_samples * B, T, -1)
        )

        # z_head: (num_samples, B, dim)
        # -> (num_samples * B, dim)
        z_head_rep = z_head.reshape(num_samples * B, -1)

        head_out = self.motion_decoder(feat_head_rep, z_head_rep)  # (num_samples*B, T, dim)

        head_betas = self.head_betas_layer(head_out)
        head_contacts = self.head_contacts_layer(head_out)
        head_theta = self.head_theta_layer(head_out)

        pose_quat = matrix_to_quaternion(
            rotation_6d_to_matrix(head_theta.reshape(num_samples * B, T, -1, 6))
        )

        return {
            "head_betas": head_betas,        # (num_samples*B, T, 16)
            "head_contacts": head_contacts,  # (num_samples*B, T, 21)
            "head_theta": head_theta,        # (num_samples*B, T, 126)
            "head_dist": dist_head,
            "head_z": z_head_rep,            # (num_samples*B, 256)
            "pose_quat": pose_quat,          # (num_samples*B, T, 21, 4)

            "motion_dist": dist_motion,
            "motion_z": z_motion,            # (B, 256)
        }