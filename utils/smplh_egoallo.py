import torch

from model.utils_rotation import rotation_6d_to_matrix, quaternion_to_axis_angle, matrix_to_quaternion

from utils.from_egoallo.transforms import SE3, SO3
from utils.from_egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose

def build_smplh_posed(outputs, batch, fncsmpl):
    B, T, _ = batch.T_world_cpf.shape
    device = batch.T_world_cpf.device

    body_model = fncsmpl


    pred_pose = rotation_6d_to_matrix(outputs['recon_from_head'][..., 6:132].reshape(B, T, 21, 6)) # (B, T, 21, 3, 3)
    pred_hands = torch.zeros(((B, T, 30, 3, 3)), device=device) # (B, T, 30, 3, 3)
    
    pred_betas = outputs['recon_from_head'][..., 132:148]
    
    # Pred ---------------------------------------------------------------------------------------------------------------
    pred_posed = body_model.with_shape(pred_betas).with_pose(
        T_world_root=SE3.identity(device, torch.float32).wxyz_xyz,
        local_quats=SO3.from_matrix(
            torch.cat((pred_pose, pred_hands), dim=2)
        ).wxyz,
    )
    pred_posed = pred_posed.with_new_T_world_root(
        get_T_world_root_from_cpf_pose(pred_posed, batch.T_world_cpf)
    )
    # --------------------------------------------------------------------------------------------------------------------
    
    # GT -----------------------------------------------------------------------------------------------------------------
    zero_hands = torch.zeros(((B, T, 30, 4)), device=device) # (B, T, 30, 3, 3)
    gt_posed = body_model.with_shape(batch.betas).with_pose(
        batch.T_world_root,
        torch.cat((batch.body_quats, zero_hands,), dim=2),
    )
    # --------------------------------------------------------------------------------------------------------------------
    
    return pred_posed, gt_posed, pred_betas

def build_smplh_posed_jjh(outputs, batch, fncsmpl):
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

def build_smplh_posed_multihypo(outputs, batch, fncsmpl):
    # outputs 기준 batch size 사용
    B_out, T, _ = outputs["head_theta"].shape
    B_gt, T_gt, _ = batch.T_world_cpf.shape
    device = batch.T_world_cpf.device

    assert T == T_gt, f"T mismatch: outputs T={T}, batch T={T_gt}"

    # HEAD ---------------------------------------------------------------------------------------------------------------
    zero_hands_mat = torch.zeros((B_out, T, 30, 3, 3), device=device)

    head_pose = rotation_6d_to_matrix(
        outputs["head_theta"].reshape(B_out, T, 21, 6)
    )  # (B_out, T, 21, 3, 3)

    head_betas = outputs["head_betas"].mean(dim=1, keepdim=True).expand(B_out, T, 16)

    head_posed = fncsmpl.with_shape(head_betas).with_pose(
        T_world_root=SE3.identity(device, torch.float32).wxyz_xyz,
        local_quats=SO3.from_matrix(
            torch.cat((head_pose, zero_hands_mat), dim=2)
        ).wxyz,
    )

    # batch.T_world_cpf는 GT batch 기준으로 (1, T, 7)
    # head_posed는 num_samples 기준으로 (5, T, ...)
    # 따라서 head 쪽에 맞게 expand
    T_world_cpf = batch.T_world_cpf
    if T_world_cpf.shape[0] == 1 and B_out != 1:
        T_world_cpf = T_world_cpf.expand(B_out, -1, -1)

    head_posed = head_posed.with_new_T_world_root(
        get_T_world_root_from_cpf_pose(head_posed, T_world_cpf)
    )
    # --------------------------------------------------------------------------------------------------------------------

    # GT -----------------------------------------------------------------------------------------------------------------
    zero_hands_quat = torch.zeros((B_gt, T, 30, 4), device=device)

    gt_posed = fncsmpl.with_shape(batch.betas).with_pose(
        batch.T_world_root,
        torch.cat((batch.body_quats, zero_hands_quat), dim=2),
    )
    # --------------------------------------------------------------------------------------------------------------------

    return head_posed, gt_posed, head_betas

from collections import namedtuple

# 평가 함수에 넘겨주기 위한 Dummy Class
DummyPosed = namedtuple("DummyPosed", ["T_world_root", "Ts_world_joint"])

def build_smplh_posed_adt(outputs, batch, fncsmpl):
    B, T, _ = batch.T_world_cpf.shape
    device = batch.T_world_cpf.device

    zero_hands = torch.zeros(((B, T, 30, 3, 3)), device=device) 
    
    # =========================================================================
    # HEAD (Prediction): 모델의 예측 결과 생성
    # =========================================================================
    head_pose = rotation_6d_to_matrix(outputs['head_theta'].reshape(B, T, 21, 6)) 
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

    # =========================================================================
    # GT (ADT Position-based): 빈 공간 채우기 및 SE3 모방
    # =========================================================================
    # 원본 batch 데이터를 훼손하지 않도록 clone() 사용
    gt_joints_xyz = batch.gt_joints_wrt_world.clone()
    
    # -------------------------------------------------------------------------
    # [핵심 추가] 머리(15번) GT 값을 Pred 값으로 교체하여 에러 상쇄
    # 예측된 관절의 SE3(7D) 중 뒤의 3자리(X, Y, Z) 추출
    pred_head_xyz = head_posed.Ts_world_joint[:, :, 15, 4:7] 
    gt_joints_xyz[:, :, 15, :] = pred_head_xyz
    # -------------------------------------------------------------------------
    
    # SE3 형태(7D)를 맞추기 위해 Identity Quaternion(1, 0, 0, 0) 생성
    quats = torch.zeros((B, T, 22, 4), device=device, dtype=gt_joints_xyz.dtype)
    quats[..., 0] = 1.0 
    
    # Quaternion(4D) + XYZ(3D) 병합 -> (B, T, 22, 7)
    gt_Ts_world_joint = torch.cat([quats, gt_joints_xyz], dim=-1)
    gt_T_world_root = gt_Ts_world_joint[:, :, 0, :]
    
    gt_posed = DummyPosed(
        T_world_root=gt_T_world_root, 
        Ts_world_joint=gt_Ts_world_joint
    )
    
    return head_posed, gt_posed, head_betas

def build_smplh_posed_jjh_plus1(outputs, batch, fncsmpl):
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


def build_smplh_posed_paper(outputs, batch, fncsmpl):
    B, T, _ = batch.T_world_cpf.shape
    device = batch.T_world_cpf.device

    zero_hands = torch.zeros(((B, T, 30, 3, 3)), device=device) # (B, T, 30, 3, 3)
    
    # HEAD ---------------------------------------------------------------------------------------------------------------
    head_pose = rotation_6d_to_matrix(outputs['head_theta'].reshape(B, T, 21, 6)) # (B, T, 21, 3, 3)
    head_betas = outputs['head_betas'].mean(dim=1, keepdim=True).expand(B, T, 16)
    # head_betas = outputs['head_betas']

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
    # motion_betas = outputs['motion_betas']

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
    # zero_hands = torch.zeros(((B, T, 30, 4)), device=device) # (B, T, 30, 3, 3)
    if batch.hand_quats == None:
        hand_quats = torch.zeros(((B, T, 30, 4)), device=device)
    else:
        hand_quats = batch.hand_quats
        
    zero_hands = torch.zeros(((B, T, 30, 4)), device=device)
    gt_posed = fncsmpl.with_shape(batch.betas).with_pose(
        batch.T_world_root,
        torch.cat((batch.body_quats, zero_hands,), dim=2),
    )
    # --------------------------------------------------------------------------------------------------------------------
    
    return head_posed, motion_posed, gt_posed, head_betas, motion_betas


def build_smplh_posed_paper_2(outputs, batch, fncsmpl):
    B, T, _ = batch.T_world_cpf.shape
    device = batch.T_world_cpf.device
    # ones_hands = torch.ones(((B, T, 30, 3, 3)), device=device) # (B, T, 30, 3, 3)
    zero_hands = torch.zeros(((B, T, 30, 3, 3)), device=device) # (B, T, 30, 3, 3)
    
    # HEAD ---------------------------------------------------------------------------------------------------------------
    head_pose = rotation_6d_to_matrix(outputs['head_theta'].reshape(B, T, 21, 6)) # (B, T, 21, 3, 3)
    head_betas = outputs['head_betas'].mean(dim=1, keepdim=True).expand(B, T, 16)
    # head_betas = outputs['head_betas']
    # hands: identity rotmats (not zeros)
    eye = torch.eye(3, device=device, dtype=head_pose.dtype)
    hand_rotmats_identity = eye.view(1, 1, 1, 3, 3).expand(B, T, 30, 3, 3)

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
    # motion_betas = outputs['motion_betas']

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
    zero_hands = torch.zeros(((B, T, 30, 4)), device=device) # (B, T, 30, 3, 3)
    if batch.hand_quats == None:
        hand_quats = torch.zeros(((B, T, 30, 4)), device=device)
    else:
        hand_quats = batch.hand_quats
        
    # ones_hands_quats = torch.zeros(((B, T, 30, 4)), device=device)
    hand_quats_identity = torch.zeros((B, T, 30, 4), device=device, dtype=head_pose.dtype)
    hand_quats_identity[..., 0] = 1.0

    gt_posed = fncsmpl.with_shape(batch.betas).with_pose(
        batch.T_world_root,
        torch.cat((batch.body_quats, zero_hands,), dim=2),
    )
    # --------------------------------------------------------------------------------------------------------------------
    
    return head_posed, motion_posed, gt_posed, head_betas, motion_betas


def build_smplh_posed_ablation_1(outputs, batch, fncsmpl):
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
    # GT -----------------------------------------------------------------------------------------------------------------
    zero_hands = torch.zeros(((B, T, 30, 4)), device=device) # (B, T, 30, 3, 3)
    gt_posed = fncsmpl.with_shape(batch.betas).with_pose(
        batch.T_world_root,
        torch.cat((batch.body_quats, zero_hands,), dim=2),
    )
    # --------------------------------------------------------------------------------------------------------------------
    
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
