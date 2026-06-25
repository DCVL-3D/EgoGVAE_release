from __future__ import annotations

import torch
from torch import Tensor
from pathlib import Path
import numpy as np

from .tensor_dataclass import TensorDataclass

# from egoallo_original.src.egoallo import fncsmpl, fncsmpl_extensions
# from egoallo_original.src.egoallo import transforms as tf

import dataclasses

class TrainingData(TensorDataclass):

    T_world_root: Tensor
    contacts: Tensor
    betas: Tensor
    body_quats: Tensor
    T_cpf_tm1_cpf_t: Tensor
    T_world_cpf: Tensor
    height_from_floor: Tensor
    joints_wrt_cpf: Tensor
    joints_wrt_world: Tensor
    mask: Tensor
    hand_quats: Tensor




class RichTestData(TensorDataclass):
    T_world_root: Tensor
    contacts: Tensor
    betas: Tensor
    body_quats: Tensor
    T_cpf_tm1_cpf_t: Tensor
    T_world_cpf: Tensor
    height_from_floor: Tensor
    joints_wrt_cpf: Tensor
    mask: Tensor
    hand_quats: Tensor
    gt_joints_wrt_world: Tensor | None = None
    @staticmethod
    def load_from_npz(
        body_model: fncsmpl.SmplhModel,
        path: Path,
        include_hands: bool,
    ) -> RichTestData:
        """Load a single trajectory from a (processed_30fps) npz file."""
        raw_fields = {
            k: torch.from_numpy(v.astype(np.float32) if v.dtype == np.float64 else v)
            for k, v in np.load(path).items()
            if v.dtype in (np.float32, np.float64)
        }

        timesteps = raw_fields["root_orient"].shape[0]
        assert raw_fields["root_orient"].shape == (timesteps, 3)
        assert raw_fields["pose_body"].shape == (timesteps, 63)
        assert raw_fields["joints"].shape == (timesteps, 22, 3)

        T_world_root = torch.cat(
            [
                tf.SO3.exp(raw_fields["root_orient"]).wxyz,
                raw_fields["joints"][:, 0, :],
            ],
            dim=-1,
        )
        body_quats = tf.SO3.exp(raw_fields["pose_body"].reshape(timesteps, 21, 3)).wxyz

        device = body_model.weights.device
        shaped = body_model.with_shape(raw_fields["betas"][0:1, :].to(device))

        # Batch the SMPL body model operations, this can be pretty memory-intensive...
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root.to(device), body_quats=body_quats.to(device)
        )
        T_world_cpf = (
            tf.SE3(posed.Ts_world_joint[:, 14, :])  # T_world_head
            @ tf.SE3(fncsmpl_extensions.get_T_head_cpf(shaped))
        ).parameters()
        assert T_world_cpf.shape == (timesteps, 7)

        # Construct the training data elements that we want to keep.
        return RichTestData(
            T_world_root=T_world_root[1:].cpu(),
            contacts=raw_fields["contacts"][1:, 1:].cpu(),  # Root is no longer a joint.
            betas=raw_fields["betas"][0:1, :].cpu(),
            body_quats=body_quats[1:].cpu(),
            T_world_cpf=T_world_cpf[1:].cpu(),
            height_from_floor=T_world_cpf[1:, 6:7].cpu(),
            T_cpf_tm1_cpf_t=(
                tf.SE3(T_world_cpf[:-1, :]).inverse() @ tf.SE3(T_world_cpf[1:, :])
            )
            .parameters()
            .cpu(),
        )



def collate_dataclass(batch: list) -> TrainingData:

    if not batch:
        raise ValueError("batch is empty")
        
    elem = batch[0]
    if not dataclasses.is_dataclass(elem):
        raise TypeError("collate_dataclass can only be used with dataclasses")
    
    collated_data = {}
    for f in dataclasses.fields(elem):
        
        key = f.name
        first_item_val = getattr(elem, key)

        if first_item_val is None:
            if all(getattr(b, key) is None for b in batch):
                collated_data[key] = None
            else:
                raise TypeError(f"Field '{key}' contains a mix of Tensors and None.")
        elif isinstance(first_item_val, torch.Tensor):
            collated_data[key] = torch.stack([getattr(b, key) for b in batch])
        else:
            collated_data[key] = [getattr(b, key) for b in batch]

    return type(elem)(**collated_data)


