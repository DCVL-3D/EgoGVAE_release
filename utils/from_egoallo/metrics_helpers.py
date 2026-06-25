from typing import Literal, overload, Optional, Dict

import numpy as np
import torch

from torch import Tensor
from typing_extensions import assert_never

from .transforms import SO3



def compute_jitter(
    Ts_world_joint: torch.Tensor,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Jitter metric based on 3rd-order finite difference (jerk),
    identical in form to your provided code.

    Args:
        Ts_world_joint: (B, T, J, 7)  where position is [:, :, :, 4:7]
        fps: frames per second (use 30 for your data)

    Returns:
        jitter_per_sample: (B,) numpy array
    """
    # Extract joint positions
    joint_pos = Ts_world_joint[..., 4:7]        # (B,T,J,3)
    B, T, J, _ = joint_pos.shape
    assert T >= 4, "Need at least 4 frames for jerk-based jitter."

    # Collapse batch and time for identical behavior to your snippet
    joint_pos = joint_pos.reshape(-1, J, 3)     # (B*T, J, 3)

    # 3rd-order finite difference (jerk)
    jerk = (
        joint_pos[3:]
        - 3 * joint_pos[2:-1]
        + 3 * joint_pos[1:-2]
        - joint_pos[:-3]
    ) * (fps ** 3)

    # L2 over xyz, mean over joints and frames
    jitter = jerk.norm(dim=2).mean(dim=1).mean() / 100.0

    # foot_skate와 동일하게 "sample-wise"로 반환하고 싶으면
    # 여기서는 batch를 다시 나누기 어렵기 때문에
    # foot_skate처럼 .mean()으로 바로 스칼라를 쓸 걸 전제로 함
    return jitter.detach().cpu().numpy()

def compute_foot_skate(
    pred_Ts_world_joint,
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]

    # Drop the person to the floor.
    # This is necessary for the foot skating metric to make sense for floating people...!
    pred_Ts_world_joint = pred_Ts_world_joint.clone()
    pred_Ts_world_joint[..., 6] -= torch.min(pred_Ts_world_joint[..., 6])

    foot_indices = torch.tensor([6, 7, 9, 10], device=pred_Ts_world_joint.device)

    foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]
    foot_positions_diff = foot_positions[:, 1:, :, :2] - foot_positions[:, :-1, :, :2]
    assert foot_positions_diff.shape == (num_samples, time - 1, 4, 2)

    foot_positions_diff_norm = torch.sum(torch.abs(foot_positions_diff), dim=-1)
    assert foot_positions_diff_norm.shape == (num_samples, time - 1, 4)

    # From EgoEgo / kinpoly.
    H_thresh = torch.tensor(
        # To match indices above: (ankle, ankle, toe, toe)
        [0.08, 0.08, 0.04, 0.04],
        device=pred_Ts_world_joint.device,
        dtype=torch.float32,
    )

    foot_positions_diff_norm = torch.sum(torch.abs(foot_positions_diff), dim=-1)
    assert foot_positions_diff_norm.shape == (num_samples, time - 1, 4)

    # Threshold.
    foot_positions_diff_norm = foot_positions_diff_norm * (
        foot_positions[..., 1:, :, 2] < H_thresh
    )
    fs_per_sample = torch.sum(
        torch.sum(
            foot_positions_diff_norm
            * (2 - 2 ** (foot_positions[..., 1:, :, 2] / H_thresh)),
            dim=-1,
        ),
        dim=-1,
    )
    assert fs_per_sample.shape == (num_samples,)

    return fs_per_sample.numpy(force=True)

from typing import Sequence, Literal
def compute_foot_skate_gt_contact_batched(
    pred_Ts_world_joint: torch.Tensor,               # (B,T,J,7)
    gt_Ts_world_joint: torch.Tensor,                 # (B,T,J,7)
    foot_indices: Sequence[int] = (6, 7, 9, 10),     # ankle, ankle, toe, toe (root 제외 기준)
    vel_thresh: float = 0.01,                        # meters / frame
    pred_vel_norm: Literal["l2", "l1"] = "l2",
    reduce: Literal["mean", "sum"] = "mean",
    to_mm: bool = True,
) -> torch.Tensor:
    """
    GT-contact-conditioned foot skating for EgoAllo-style tensors.

    pred_Ts_world_joint, gt_Ts_world_joint: (B,T,J,7)
      - position: [..., 4:7]
      - z:        [..., 6]

    Returns:
      skate_per_sample: (B,) torch tensor (mm if to_mm=True else meters)
    """
    assert pred_Ts_world_joint.shape == gt_Ts_world_joint.shape, \
        f"Shape mismatch: pred {tuple(pred_Ts_world_joint.shape)} vs gt {tuple(gt_Ts_world_joint.shape)}"
    assert pred_Ts_world_joint.ndim == 4 and pred_Ts_world_joint.shape[-1] == 7, \
        f"Expected (B,T,J,7), got {tuple(pred_Ts_world_joint.shape)}"

    B, T, J, _ = pred_Ts_world_joint.shape
    assert T >= 2, "Need at least 2 frames."

    device = pred_Ts_world_joint.device
    idx = torch.as_tensor(list(foot_indices), device=device, dtype=torch.long)

    # Extract positions (B,T,J,3)
    pred_pos = pred_Ts_world_joint[..., 4:7]
    gt_pos   = gt_Ts_world_joint[..., 4:7]

    # Select foot joints: (B,T,4,3)
    pred_foot = pred_pos.index_select(dim=2, index=idx)
    gt_foot   = gt_pos.index_select(dim=2, index=idx)

    # Velocities: (B,T-1,4,3)
    gt_vel   = gt_foot[:, 1:]   - gt_foot[:, :-1]
    pred_vel = pred_foot[:, 1:] - pred_foot[:, :-1]

    # Speeds: (B,T-1,4)
    if pred_vel_norm == "l2":
        gt_speed   = torch.linalg.norm(gt_vel, dim=-1)
        pred_speed = torch.linalg.norm(pred_vel, dim=-1)
    else:
        gt_speed   = torch.sum(torch.abs(gt_vel), dim=-1)
        pred_speed = torch.sum(torch.abs(pred_vel), dim=-1)

    # Contact mask from GT: (B,T-1,4)
    contact = (gt_speed <= vel_thresh)

    # Penalize pred speed only where GT says contact
    masked = pred_speed * contact.to(pred_speed.dtype)  # (B,T-1,4)

    # Reduce per sample -> (B,)
    if reduce == "mean":
        skate = masked.mean(dim=(1, 2))   # mean over time*feet (비접촉은 0 포함)
    else:
        skate = masked.sum(dim=(1, 2))

    if to_mm:
        skate = skate * 1000.0

    return skate

# def compute_foot_skate(
#     pred_Ts_world_joint: torch.Tensor,
#     gt_Ts_world_joint: torch.Tensor,
#     foot_joint_indices = (6, 7, 9, 10),  # L_Ankle, R_Ankle, L_Toe, R_Toe (네 정의에 맞게)
#     contact_vel_thresh_m: float = 0.01,                  # GT speed threshold in meters/frame
#     dt: float = 1.0,                                     # frame interval. keep 1.0 => per-frame
# ) -> np.ndarray:
#     """
#     GT-contact-conditioned foot skating metric.

#     1) Define contact frames using GT foot speed:
#          contact(t) = ||p_gt(t+1) - p_gt(t)||_2 / dt <= thresh
#     2) Penalize predicted foot velocity magnitude on those contact frames:
#          skate = mean_{contact frames, foot joints} ||p_pred(t+1)-p_pred(t)||_2 / dt

#     Returns:
#         (B,) numpy array in mm/frame (if dt=1.0).
#     """

#     pred = pred_Ts_world_joint
#     gt = gt_Ts_world_joint

#     B, T, J, D = pred.shape
#     assert gt.shape[:3] == (B, T, gt.shape[2]), f"GT shape mismatch: pred {pred.shape}, gt {gt.shape}"
#     assert D >= 7, f"Expected last dim >=7, got {D}"
#     assert T >= 2, "Need at least 2 frames to compute velocities."

#     device = pred.device
#     idx = torch.as_tensor(list(foot_joint_indices), device=device, dtype=torch.long)

#     # positions (meters)
#     pred_pos = pred[..., 4:7]  # (B,T,J,3)
#     gt_pos   = gt[..., 4:7]    # (B,T,J,3)

#     pred_foot = pred_pos.index_select(dim=2, index=idx)  # (B,T,F,3)
#     gt_foot   = gt_pos.index_select(dim=2, index=idx)    # (B,T,F,3)

#     # velocities (meters/dt)
#     pred_vel = (pred_foot[:, 1:] - pred_foot[:, :-1]) / dt  # (B,T-1,F,3)
#     gt_vel   = (gt_foot[:, 1:]   - gt_foot[:, :-1])   / dt  # (B,T-1,F,3)

#     # GT contact mask based on speed
#     gt_speed = torch.linalg.norm(gt_vel, dim=-1)  # (B,T-1,F)
#     contact_mask = (gt_speed <= contact_vel_thresh_m / dt)  # (B,T-1,F)

#     # Penalize predicted speed magnitude on contact frames
#     pred_speed = torch.linalg.norm(pred_vel, dim=-1)  # (B,T-1,F)

#     # avoid boolean indexing flattening: masked sum / count
#     pred_speed_contact = pred_speed * contact_mask.to(pred_speed.dtype)  # zeros where not contact
#     denom = contact_mask.sum(dim=(1, 2)).clamp_min(1)                    # (B,)
#     skate_m_per_dt = pred_speed_contact.sum(dim=(1, 2)) / denom          # (B,)

#     # convert to mm/(dt unit). dt=1 => mm/frame
#     skate_mm = skate_m_per_dt * 1000.0

#     return skate_mm.detach().cpu().numpy()


def compute_floating_error(
    pred_Ts_world_joint: torch.Tensor,
    floor_height: torch.Tensor,
    foot_indices=(6, 7, 9, 10),  # ankle/ankle/toe/toe (네 인덱스 체계에 맞춰 조정)
) -> np.ndarray:
    """
    Floating metric using per-frame floor height.

    Args:
        pred_Ts_world_joint: (B, T, J, 7), position is [..., 4:7] and z is [..., 6]
        floor_height:
            - (B, T) or (B, T, 1)  : per-frame height reference (e.g., batch.T_world_cpf[..., 6:7])
            - (B,)                 : per-sequence height reference (will be expanded)
        foot_indices: joints to estimate lowest foot height per frame

    Returns:
        floating_per_sample: (B,) numpy array
    """
    B, T = pred_Ts_world_joint.shape[:2]
    # (B,T,J) z
    z = pred_Ts_world_joint[..., 6]

    # lowest foot z per frame: (B,T)
    foot_idx = torch.as_tensor(foot_indices, device=pred_Ts_world_joint.device)
    z_sel = z.index_select(dim=2, index=foot_idx)          # (B,T,|F|)
    z_min = z_sel.min(dim=2).values                        # (B,T)

    # floor_height -> (B,T)
    # print(floor_height)
    fh = floor_height.to(device=pred_Ts_world_joint.device, dtype=pred_Ts_world_joint.dtype).squeeze(-1)
    print(fh.shape)
    if fh.ndim == 1:
        # (B,) -> (B,T)
        assert fh.shape[0] == B
        fh = fh[:, None].expand(B, T)
    else:
        # (B,T)
        assert fh.shape[0] == B and fh.shape[1] == T

    # floating only: max(0, z_min - fh)
    floating = torch.relu(z_min - fh).mean(dim=1)          # (B,)

    floating = floating * 1000.0

    return floating.detach().cpu().numpy()

import torch
from typing import Sequence, Literal, Tuple

def ground_metrics_lowest_egoallo(
    pred_Ts_world_joint: torch.Tensor,   # (B,T,J,7)
    gt_Ts_world_joint: torch.Tensor,     # (B,T,J,7)
    joint_indices: Sequence[int] | None = None,  # None = 전체 joints
    unit: Literal["m", "cm", "mm"] = "mm",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EgoAllo-style Ground / Penetration / Floating metric.

    Returns:
        ground, penetration, floating : (B,) tensor
    """
    assert pred_Ts_world_joint.shape == gt_Ts_world_joint.shape
    assert pred_Ts_world_joint.ndim == 4 and pred_Ts_world_joint.shape[-1] == 7

    B, T, J, _ = pred_Ts_world_joint.shape

    # (B,T,J,3)
    pred_pos = pred_Ts_world_joint[..., 4:7]
    gt_pos   = gt_Ts_world_joint[..., 4:7]

    # optionally restrict joints
    if joint_indices is not None:
        idx = torch.as_tensor(joint_indices, device=pred_pos.device, dtype=torch.long)
        pred_pos = pred_pos.index_select(dim=2, index=idx)
        gt_pos   = gt_pos.index_select(dim=2, index=idx)

    # lowest joint z per frame
    min_z_pred = pred_pos[..., 2].min(dim=2).values   # (B,T)
    min_z_gt   = gt_pos[..., 2].min(dim=2).values     # (B,T)

    # per-sample metrics (meters)
    penetration = torch.relu(min_z_gt - min_z_pred).mean(dim=1)  # (B,)
    floating    = torch.relu(min_z_pred - min_z_gt).mean(dim=1)  # (B,)
    ground      = penetration + floating                         # (B,)

    # unit conversion
    scale = 1.0
    if unit == "cm":
        scale = 100.0
    elif unit == "mm":
        scale = 1000.0

    penetration = penetration * scale
    floating = floating * scale
    return penetration.detach().cpu().numpy().mean(), floating.detach().cpu().numpy().mean()


def compute_foot_contact(
    pred_Ts_world_joint,
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]

    foot_indices = torch.tensor([6, 7, 9, 10], device=pred_Ts_world_joint.device)

    # From EgoEgo / kinpoly.
    H_thresh = torch.tensor(
        # To match indices above: (ankle, ankle, toe, toe)
        [0.08, 0.08, 0.04, 0.04],
        device=pred_Ts_world_joint.device,
        dtype=torch.float32,
    )

    foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]

    any_contact = torch.any(
        torch.any(foot_positions[..., 2] < H_thresh, dim=-1), dim=-1
    ).to(torch.float32)
    assert any_contact.shape == (num_samples,)

    return any_contact.numpy(force=True)


def compute_head_ori(
    label_Ts_world_joint,
    pred_Ts_world_joint,
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]
    matrix_errors = (
        SO3(pred_Ts_world_joint[:, :, 14, :4]).as_matrix()
        @ SO3(label_Ts_world_joint[:, 14, :4]).inverse().as_matrix()
    ) - torch.eye(3, device=label_Ts_world_joint.device)
    assert matrix_errors.shape == (num_samples, time, 3, 3)

    return torch.mean(
        torch.linalg.norm(matrix_errors.reshape((num_samples, time, 9)), dim=-1),
        dim=-1,
    ).numpy(force=True)


def compute_head_trans(
    label_Ts_world_joint,
    pred_Ts_world_joint,
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]
    errors = pred_Ts_world_joint[:, :, 14, 4:7] - label_Ts_world_joint[:, :, 14, 4:7]
    assert errors.shape == (num_samples, time, 3)

    return torch.mean(
        torch.linalg.norm(errors, dim=-1),
        dim=-1,
    ).numpy(force=True)


def compute_mpjpe(
    label_T_world_root,
    label_Ts_world_joint,
    pred_T_world_root,
    pred_Ts_world_joint,
    per_frame_procrustes_align: bool,
) -> np.ndarray:
    num_samples, time, _, _ = pred_Ts_world_joint.shape

    # root_pts1 = torch.zeros_like(label_T_world_root[..., None, :]) # (B, T, 1, 7)
    # root_pts2 = torch.zeros_like(pred_T_world_root[..., None, :])  # (B, T, 1, 7)
    # Concatenate the world root to the joints.
    label_Ts_world_joint = torch.cat(
        [label_T_world_root[..., None, :], label_Ts_world_joint], dim=-2
    )
    pred_Ts_world_joint = torch.cat(
        [pred_T_world_root[..., None, :], pred_Ts_world_joint], dim=-2
    )
    del label_T_world_root, pred_T_world_root
    
    # print(f"label_T_world_root{label_T_world_root[..., None, :]}")
    # print(f"pred_T_world_root{pred_T_world_root[..., None, :]}")
    
    # label_Ts_world_joint = torch.cat(
    #     [root_pts1, label_Ts_world_joint], dim=-2
    # )
    # pred_Ts_world_joint = torch.cat(
    #     [root_pts2, pred_Ts_world_joint], dim=-2
    # )
    # del label_T_world_root, pred_T_world_root

    pred_joint_positions = pred_Ts_world_joint[:, :, :, 4:7]
    label_joint_positions = label_Ts_world_joint[:, :, :, 4:7]

    if per_frame_procrustes_align:
        pred_joint_positions = procrustes_align(
            points_y=pred_joint_positions,
            points_x=label_joint_positions,
            output="aligned_x",
        )
    
    position_differences = pred_joint_positions - label_joint_positions
    assert position_differences.shape == (num_samples, time, 22, 3)

    # Per-joint position errors, in millimeters.
    pjpe = torch.linalg.norm(position_differences, dim=-1) * 1000.0
    assert pjpe.shape == (num_samples, time, 22)

    # Mean per-joint position errors.
    mpjpe = torch.mean(pjpe.reshape((num_samples, -1)), dim=-1)
    assert mpjpe.shape == (num_samples,)

    return mpjpe.cpu().numpy()

def compute_mpjpe_train(
    label_T_world_root,
    label_Ts_world_joint,
    pred_T_world_root,
    pred_Ts_world_joint,
    per_frame_procrustes_align: bool,
) -> np.ndarray:
    num_samples, time, _, _ = pred_Ts_world_joint.shape

    # root_pts1 = torch.zeros_like(label_T_world_root[..., None, :]) # (B, T, 1, 7)
    # root_pts2 = torch.zeros_like(pred_T_world_root[..., None, :])  # (B, T, 1, 7)
    # Concatenate the world root to the joints.
    label_Ts_world_joint = torch.cat(
        [label_T_world_root[..., None, :], label_Ts_world_joint], dim=-2
    )
    pred_Ts_world_joint = torch.cat(
        [pred_T_world_root[..., None, :], pred_Ts_world_joint], dim=-2
    )
    del label_T_world_root, pred_T_world_root
    # print(f"label_T_world_root{label_T_world_root[..., None, :]}")
    # print(f"pred_T_world_root{pred_T_world_root[..., None, :]}")
    
    # label_Ts_world_joint = torch.cat(
    #     [root_pts1, label_Ts_world_joint], dim=-2
    # )
    # pred_Ts_world_joint = torch.cat(
    #     [root_pts2, pred_Ts_world_joint], dim=-2
    # )
    # del label_T_world_root, pred_T_world_root

    pred_joint_positions = pred_Ts_world_joint[:, :, :, 4:7]
    label_joint_positions = label_Ts_world_joint[:, :, :, 4:7]

    if per_frame_procrustes_align:
        pred_joint_positions = procrustes_align(
            points_y=pred_joint_positions,
            points_x=label_joint_positions,
            output="aligned_x",
        )
    
    position_differences = pred_joint_positions - label_joint_positions
    assert position_differences.shape == (num_samples, time, 22, 3)

    # Per-joint position errors, in millimeters.
    pjpe = torch.linalg.norm(position_differences, dim=-1) * 1000.0
    assert pjpe.shape == (num_samples, time, 22)

    # Mean per-joint position errors.
    mpjpe = torch.mean(pjpe.reshape((num_samples, -1)), dim=-1)
    assert mpjpe.shape == (num_samples,)

    return mpjpe.detach().cpu().numpy()

def compute_mpjve(
    label_T_world_root,
    label_Ts_world_joint,
    pred_T_world_root,
    pred_Ts_world_joint,
    per_frame_procrustes_align: bool,
    fps: float = 30.0,   # 프레임 간 시간 간격(초). 모르면 1.0으로 두면 "per-frame" 속도오차
) -> np.ndarray:
    """
    MPJVE (Mean Per Joint Velocity Error)
    - compute_mpjpe와 동일하게 root를 joints 앞에 concat해서 (num_samples, time, 22, 3)로 맞춘 뒤,
      joint velocity의 차이를 L2로 재고(mm/s 또는 mm/frame) 평균냄.
    - 반환: (num_samples,) numpy
    """
    num_samples, time, _, _ = pred_Ts_world_joint.shape

    # Concatenate the world root to the joints (mpjpe와 동일)
    label_Ts_world_joint = torch.cat(
        [label_T_world_root[..., None, :], label_Ts_world_joint], dim=-2
    )
    pred_Ts_world_joint = torch.cat(
        [pred_T_world_root[..., None, :], pred_Ts_world_joint], dim=-2
    )
    del label_T_world_root, pred_T_world_root

    pred_pos = pred_Ts_world_joint[:, :, :, 4:7]   # (B,T,22,3)
    label_pos = label_Ts_world_joint[:, :, :, 4:7] # (B,T,22,3)

    if per_frame_procrustes_align:
        pred_pos = procrustes_align(
            points_y=pred_pos,
            points_x=label_pos,
            output="aligned_x",
        )

    # velocities: (B, T-1, 22, 3)
    pred_vel = (pred_pos[:, 1:] - pred_pos[:, :-1]) * fps
    label_vel = (label_pos[:, 1:] - label_pos[:, :-1]) * fps

    vel_diff = pred_vel - label_vel
    assert vel_diff.shape == (num_samples, time - 1, 22, 3)

    # Per-joint velocity errors -> millimeters/(dt unit)
    pjve = torch.linalg.norm(vel_diff, dim=-1) * 1000.0  # (B,T-1,22)
    assert pjve.shape == (num_samples, time - 1, 22)

    mpjve = torch.mean(pjve.reshape((num_samples, -1)), dim=-1)  # (B,)
    assert mpjve.shape == (num_samples,)

    return mpjve.detach().cpu().numpy()


@overload
def procrustes_align(
    points_y,
    points_x,
    output,
    fix_scale: bool = False,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def procrustes_align(
    points_y,
    points_x,
    output,
    fix_scale: bool = False,
) -> Tensor: ...


def procrustes_align(
    points_y,
    points_x,
    output: Literal["transforms", "aligned_x"],
    fix_scale: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | Tensor:
    """Similarity transform alignment using the Umeyama method. Adapted from
    SLAHMR: https://github.com/vye16/slahmr/blob/main/slahmr/geometry/pcl.py
    Minimizes:
        mean( || Y - s * (R @ X) + t ||^2 )
    with respect to s, R, and t.
    Returns an (s, R, t) tuple.
    """
    *dims, N, _ = points_y.shape
    device = points_y.device
    N = torch.ones((*dims, 1, 1), device=device) * N

    # subtract mean
    my = points_y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = points_x.sum(dim=-2) / N[..., 0]
    y0 = points_y - my[..., None, :]  # (*, N, 3)
    x0 = points_x - mx[..., None, :]

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = (
        torch.eye(3, device=device)
        .reshape(*(1,) * (len(dims)), 3, 3)
        .repeat(*dims, 1, 1)
    )
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S = torch.where(
        neg.reshape(*dims, 1, 1),
        S * torch.diag(torch.tensor([1, 1, -1], device=device)),
        S,
    )

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fix_scale:
        s = torch.ones(*dims, 1, device=device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = (
            torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(
                dim=-1, keepdim=True
            )
            / var[..., 0]
        )  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    assert s.shape == (*dims, 1)
    assert R.shape == (*dims, 3, 3)
    assert t.shape == (*dims, 3)

    if output == "transforms":
        return s, R, t
    elif output == "aligned_x":
        aligned_x = (
            s[..., None, :] * torch.einsum("...ij,...nj->...ni", R, points_x)
            + t[..., None, :]
        )
        assert aligned_x.shape == points_x.shape
        return aligned_x
    else:
        assert_never(output)
