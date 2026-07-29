import torch

from model.utils_rotation import rotation_6d_to_matrix, quaternion_to_axis_angle

from utils.from_egoallo.transforms import SE3, SO3
from utils.from_egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose

def build_smplh_posed(outputs, batch, fncsmpl):
    B, T, _ = batch.T_world_cpf.shape
    device = batch.T_world_cpf.device

    zero_hands = torch.zeros(((B, T, 30, 3, 3)), device=device) # (B, T, 30, 3, 3)
    
    # HEAD ---------------------------------------------------------------------------------------------------------------
    head_pose = rotation_6d_to_matrix(outputs['head_theta'].reshape(B, T, 21, 6)) # (B, T, 21, 3, 3)
    head_betas = outputs['head_betas'].mean(dim=1, keepdim=True).expand(B, T, 16)

    head_posed = fncsmpl.with_shape(head_betas).with_pose(
        T_world_root=SE3.identity(device, torch.float32).wxyz_xyz,
        local_quats=SO3.from_matrix(
            torch.cat((head_pose, zero_hands), dim=2)
        ).wxyz,
    )
    head_posed = head_posed.with_new_T_world_root(
        get_T_world_root_from_cpf_pose(head_posed, batch.T_world_cpf)
    )
    # --------------------------------------------------------------------------------------------------------------------

    # MOTION ---------------------------------------------------------------------------------------------------------------
    motion_pose = rotation_6d_to_matrix(outputs['motion_theta'].reshape(B, T, 21, 6)) # (B, T, 21, 3, 3)
    motion_betas = outputs['motion_betas'].mean(dim=1, keepdim=True).expand(B, T, 16)

    motion_posed = fncsmpl.with_shape(motion_betas).with_pose(
        T_world_root=SE3.identity(device, torch.float32).wxyz_xyz,
        local_quats=SO3.from_matrix(
            torch.cat((motion_pose, zero_hands), dim=2)
        ).wxyz,
    )
    motion_posed = motion_posed.with_new_T_world_root(
        get_T_world_root_from_cpf_pose(motion_posed, batch.T_world_cpf)
    )
    # --------------------------------------------------------------------------------------------------------------------

    # GT -----------------------------------------------------------------------------------------------------------------
    zero_hands = torch.zeros(((B, T, 30, 4)), device=device)
    gt_posed = fncsmpl.with_shape(batch.betas).with_pose(
        batch.T_world_root,
        torch.cat((batch.body_quats, zero_hands,), dim=2),
    )

    return head_posed, motion_posed, gt_posed, head_betas, motion_betas

def evaluation_build_smplh_posed(outputs, batch, fncsmpl):
    B, T, _ = batch.T_world_cpf.shape
    device = batch.T_world_cpf.device

    zero_hands = torch.zeros(((B, T, 30, 3, 3)), device=device) # (B, T, 30, 3, 3)
    
    # HEAD ---------------------------------------------------------------------------------------------------------------
    head_pose = rotation_6d_to_matrix(outputs['head_theta'].reshape(B, T, 21, 6)) # (B, T, 21, 3, 3)
    head_betas = outputs['head_betas'].mean(dim=1, keepdim=True).expand(B, T, 16)
    
    head_posed = fncsmpl.with_shape(head_betas).with_pose(
        T_world_root=SE3.identity(device, torch.float32).wxyz_xyz,
        local_quats=SO3.from_matrix(
            torch.cat((head_pose, zero_hands), dim=2)
        ).wxyz,
    )
    head_posed = head_posed.with_new_T_world_root(
        get_T_world_root_from_cpf_pose(head_posed, batch.T_world_cpf)
    )
    # --------------------------------------------------------------------------------------------------------------------

    if batch.hand_quats == None:
        hand_quats = torch.zeros(((B, T, 30, 4)), device=device)
    else:
        hand_quats = batch.hand_quats

    zero_hands = torch.zeros(((B, T, 30, 4)), device=device)
    # GT -----------------------------------------------------------------------------------------------------------------
    gt_posed = fncsmpl.with_shape(batch.betas).with_pose(
        batch.T_world_root,
        torch.cat((batch.body_quats, zero_hands,), dim=2),
    )
    # --------------------------------------------------------------------------------------------------------------------
    
    return head_posed, gt_posed, head_betas

def build_smplh_posed_egoallo(outputs, batch, fncsmpl):
    """
    For the case where batch contains Tfull frames (subseq_len+1),
    but the model outputs correspond to frames 1..Tfull-1 (length T = Tfull-1).
    """
    device = batch.T_world_cpf.device
    dtype = torch.float32

    # Model time length (safe source of truth)
    B = outputs["head_theta"].shape[0]
    T = outputs["head_theta"].shape[1]

    # ---------------------------
    # HEAD pred posed (frames 1:)
    # ---------------------------
    head_pose = rotation_6d_to_matrix(outputs["head_theta"].reshape(B, T, 21, 6))  # (B,T,21,3,3)

    # betas: make (B,T,16)
    head_betas = outputs["head_betas"]
    if head_betas.dim() == 2:  # (B,16)
        head_betas = head_betas[:, None, :].expand(B, T, 16)
    elif head_betas.dim() == 3:  # (B,T,16)
        head_betas = head_betas.mean(dim=1, keepdim=True).expand(B, T, 16)
    else:
        raise ValueError(f"Unexpected head_betas shape: {head_betas.shape}")

    # hands: identity rotmats (not zeros)
    eye = torch.eye(3, device=device, dtype=head_pose.dtype)
    hand_rotmats = eye.view(1, 1, 1, 3, 3).expand(B, T, 30, 3, 3)

    head_posed = fncsmpl.with_shape(head_betas).with_pose(
        T_world_root=SE3.identity(device, dtype).wxyz_xyz,
        local_quats=SO3.from_matrix(torch.cat((head_pose, hand_rotmats), dim=2)).wxyz,
    )

    # IMPORTANT: use batch.T_world_cpf[:, 1:] to align with T
    head_posed = head_posed.with_new_T_world_root(
        get_T_world_root_from_cpf_pose(head_posed, batch.T_world_cpf[:, 1:, :])
    )

    # ---------------------------
    # GT posed (frames 1:)
    # ---------------------------
    # betas in your data are usually (B,1,16) or (1,16). Make (B,T,16) if needed.
    gt_betas = batch.betas
    if gt_betas.dim() == 2:  # (1,16) or (B,16)
        if gt_betas.shape[0] == 1 and B > 1:
            gt_betas = gt_betas.expand(B, -1)
        gt_betas = gt_betas[:, None, :].expand(B, T, 16)
    elif gt_betas.dim() == 3:  # (B,Tfull,16) or (B,1,16)
        if gt_betas.shape[1] == 1:
            gt_betas = gt_betas.expand(B, T, 16)
        else:
            gt_betas = gt_betas[:, 1:1+T, :]
    else:
        raise ValueError(f"Unexpected batch.betas shape: {batch.betas.shape}")

    # body_quats: (B,Tfull,21,4) -> (B,T,21,4)
    gt_body_quats = batch.body_quats[:, 1:1+T, ...]

    # hands quats: if absent, identity in quat form (w=1)
    if getattr(batch, "hand_quats", None) is None:
        gt_hand_quats = torch.zeros((B, T, 30, 4), device=device, dtype=gt_body_quats.dtype)
        gt_hand_quats[..., 0] = 1.0
    else:
        gt_hand_quats = batch.hand_quats[:, 1:1+T, ...]

    gt_posed = fncsmpl.with_shape(gt_betas).with_pose(
        batch.T_world_root[:, 1:1+T, :],
        torch.cat((gt_body_quats, gt_hand_quats), dim=2),
    )

    return head_posed, gt_posed, head_betas


def extract_joints_vertices(smpl_posed, betas, smplh):
    B, T, _ = smpl_posed.T_world_root.shape

    if betas.shape[1] == 1:
        betas = betas.expand(B, T, -1)

    global_orient_aa = quaternion_to_axis_angle(smpl_posed.T_world_root[..., :4].reshape(B * T, 4)).squeeze(1)
    body_pose_aa = quaternion_to_axis_angle(smpl_posed.local_quats[:, :, :21].reshape(B * T, -1, 4)).reshape(B * T, -1)
    # betas = betas.reshape(B*T, -1)[:, :10]
    betas = betas.reshape(B*T, -1)
    smpl_input_params = {
        'global_orient': global_orient_aa,
        'body_pose': body_pose_aa,
        'betas': betas,
        'transl': (smpl_posed.T_world_root[..., 4:]- smpl_posed.T_world_root[:, [0], 4:]).reshape(B*T, -1),
    }

    smpl_output = smplh(**{k: v for k,v in smpl_input_params.items()}, pose2rot=True)
    vertices = smpl_output.vertices.reshape(B, T, 6890, 3)  
    joints = smpl_output.joints[:, :22, :].reshape(B, T, 22, 3)
    
    return joints, vertices
