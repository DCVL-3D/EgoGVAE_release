import dataclasses
from dataclasses import dataclass
import torch

import model.utils_rotation as ru
from utils.from_egoallo.transforms import SE3, SO3
from utils.from_egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose
from model.utils_rotation import rotation_6d_to_matrix

@dataclass
class PostGuidanceTraj:
    betas: torch.Tensor        # (1,T,16)
    body_rotmats: torch.Tensor # (1,T,21,3,3)
    contacts: torch.Tensor     # (1,T,21)
    hand_rotmats: torch.Tensor # (1,T,30,3,3)


def _I_hand_rotmats(B: int, T: int, device, dtype):
    return torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3).repeat(B, T, 30, 1, 1)


def build_head_posed_with_post_guidance(
    outputs: dict,
    batch,
    fncsmpl,
    guidance_mode,
    do_guidance_optimization,
    guidance_verbose: bool = False,
):
    """
    returns:
      head_posed_guided, head_betas
    """
    # B, T, _ = batch.T_world_cpf.shape
    Ts_world_cpf_used = batch.T_world_cpf[:, 1:, :]   # (128, 7)  (B=1이면)
    B, T, _ = Ts_world_cpf_used.shape
    assert B == 1, "This implementation assumes batch size = 1."

    device = batch.T_world_cpf.device
    dtype = torch.float32  # 너 build 함수도 float32로 쓰고 있어서 맞춤

    # --- 1) (초기) body rotmats 만들기: outputs["pose_quat"] 기반이 제일 안전
    # pose_quat: (1,T,21,4) wxyz
    # body_rotmats_init = ru.quaternion_to_matrix(outputs["pose_quat"]).to(dtype)
    # print(body_rotmats_init.shape)
    body_rotmats_init = rotation_6d_to_matrix(outputs['head_theta'].reshape(B, T, 21, 6)) # (B, T, 21, 3, 3)

    # --- 2) betas / contacts 준비 (EgoAllo 최적화는 betas timestep axis가 있어도 OK, 내부에서 mean씀)
    betas = outputs["head_betas"].to(dtype)        # (1,T,16)
    contacts = outputs["head_contacts"].to(dtype)  # (1,T,21)

    # --- 3) hand rotmats는 identity로 채움 (EgoAllo JAX solver가 51 joints 가정)
    hand_rotmats_I = _I_hand_rotmats(B, T, device=device, dtype=dtype)

    traj = PostGuidanceTraj(
        betas=betas,
        body_rotmats=body_rotmats_init,
        contacts=contacts,
        hand_rotmats=hand_rotmats_I,
    )
    # traj.contacts.zero_()
    # --- 4) Ts_world_cpf: (T,7)
    Ts_world_cpf = batch.T_world_cpf[0, 1:, :].to(dtype)  # (T,7)

    # --- 5) post-guidance (HaMeR/Aria 없음)
    traj_post, _ = do_guidance_optimization(
        Ts_world_cpf=Ts_world_cpf,
        traj=traj,
        body_model=fncsmpl,
        guidance_mode=guidance_mode,
        phase="post",
        hamer_detections=None,
        aria_detections=None,
        verbose=guidance_verbose,
    )

    # --- 6) traj_post.body_rotmats로 head_posed 만들기 (6D로 되돌릴 필요 없음)
    head_betas = betas.mean(dim=1, keepdim=True).expand(B, T, 16)  # 네 기존 로직 유지

    head_posed = fncsmpl.with_shape(head_betas).with_pose(
        T_world_root=SE3.identity(device, dtype).wxyz_xyz,
        local_quats=SO3.from_matrix(
            torch.cat((traj_post.body_rotmats.to(dtype), hand_rotmats_I), dim=2)  # (B,T,51,3,3)
        ).wxyz,
    )

    # head trajectory에 맞춰 root 세팅 (너 기존 방식)
    head_posed = head_posed.with_new_T_world_root(
        get_T_world_root_from_cpf_pose(head_posed, batch.T_world_cpf[:, 1:, :])
    )

    return head_posed, head_betas
