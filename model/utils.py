import torch
import cv2
import numpy as np
import random

import model.utils_rotation as ru

from utils.from_egoallo.transforms import SE3, SO3
from data.dataclass import RichTestData

def input_data_cvpr(batch):
    B, T, _ = batch.T_world_cpf.shape
    # print(batch.T_world_cpf.shape)

    #####################################################################################################################################
    height_from_floor = batch.T_world_cpf[..., 6:7]
    #####################################################################################################################################
    T_cpf_tm1_cpf_t = SE3(batch.T_cpf_tm1_cpf_t).as_matrix()[..., :3, :]
    T_cpf_tm1_cpf_t_R = ru.matrix_to_rotation_6d(T_cpf_tm1_cpf_t[..., :3]) # (B, T, 6)
    T_cpf_tm1_cpf_t_P = T_cpf_tm1_cpf_t[..., 3:].squeeze(-1) # (B, T, 3)
    
    cond_parts = [
        T_cpf_tm1_cpf_t_R,
        T_cpf_tm1_cpf_t_P,
        height_from_floor,
    ]

    R_world_cpf = SE3(batch.T_world_cpf).rotation().wxyz
    forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
    forward_world = SO3(R_world_cpf) @ forward_cpf
    assert forward_world.shape == (B, T, 3)
    R_canonical_world = SO3.from_z_radians(
        -torch.arctan2(forward_world[..., 1], forward_world[..., 0])
    ).wxyz
    assert R_canonical_world.shape == (B, T, 4)
    cond_parts.append(
        ru.matrix_to_rotation_6d((SO3(R_canonical_world) @ SO3(R_world_cpf)).as_matrix()),
    )

    head = torch.cat(cond_parts, dim=-1) # (B, T, 16)
    #####################################################################################################################################
    # global_orient = ru.quaternion_to_matrix(batch.T_world_root[..., :4]).unsqueeze(2) # [B, T, 1, 3, 3]
    body_pose = ru.quaternion_to_matrix(batch.body_quats) # [B, T, 21, 3, 3]
    # root_trans = batch.T_world_root[..., 4:] # [B, T, 3]
    
    betas = batch.betas # [B, T, 16]
    contacts = batch.contacts # [B, T, 21]

    local_rotation_body = ru.matrix_to_rotation_6d(body_pose).reshape(B, T, -1) # [B, T, 126]

    motion = torch.cat((local_rotation_body, betas, contacts), dim=-1) # [B, T, 126 + 16 + 21]
    # print(motion.shape)
    #####################################################################################################################################

    return head, motion   

def input_data(batch):

    B, T, _ = batch.T_world_cpf.shape

    head_R = ru.matrix_to_rotation_6d(ru.quaternion_to_matrix(batch.T_world_cpf[..., :4])) # [B, T, 6]
    head_T = batch.T_world_cpf[..., 4:] # [B, T, 3]
    
    head_delta_R = ru.matrix_to_rotation_6d(ru.quaternion_to_matrix(batch.T_cpf_tm1_cpf_t[..., :4])) # [B, T, 6]
    head_delta_T = batch.T_cpf_tm1_cpf_t[..., 4:] # [B, T, 3]
    head = torch.cat([head_R, head_T], dim=-1) # [B, T, 9]
    # head = torch.cat([head_R, head_T, head_delta_R, head_delta_T], dim=-1) # [B, T, 18]
    
    
    global_orient = ru.quaternion_to_matrix(batch.T_world_root[..., :4]).unsqueeze(2) # [B, T, 1, 3, 3]
    body_pose = ru.quaternion_to_matrix(batch.body_quats) # [B, T, 21, 3, 3]
    root_trans = batch.T_world_root[..., 4:] # [B, T, 3]
    
    local_rotation_body = ru.matrix_to_rotation_6d(torch.cat((global_orient, body_pose), dim=2)).reshape(B, T, -1) # [B, T, 132]
    motion = torch.cat((local_rotation_body, root_trans), dim=-1) # [B, T, 135]

    return head, motion

def input_data_egoallo(batch):
    B, T, _ = batch.T_world_cpf.shape # batch, T, 7
    # print(batch.T_world_cpf.shape) 
    #####################################################################################################################################
    height_from_floor = batch.T_world_cpf[..., 6:7]
    #####################################################################################################################################
    # print(batch.T_cpf_tm1_cpf_t.shape)
    T_cpf_tm1_cpf_t = SE3(batch.T_cpf_tm1_cpf_t).as_matrix()[..., :3, :]
    T_cpf_tm1_cpf_t_R = ru.matrix_to_rotation_6d(T_cpf_tm1_cpf_t[..., :3]) # (B, T, 6)
    T_cpf_tm1_cpf_t_P = T_cpf_tm1_cpf_t[..., 3:].squeeze(-1) # (B, T, 3)
    
    cond_parts = [
        T_cpf_tm1_cpf_t_R,
        T_cpf_tm1_cpf_t_P,
        height_from_floor,
    ]

    R_world_cpf = SE3(batch.T_world_cpf).rotation().wxyz
    forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
    forward_world = SO3(R_world_cpf) @ forward_cpf
    assert forward_world.shape == (B, T, 3)
    R_canonical_world = SO3.from_z_radians(
        -torch.arctan2(forward_world[..., 1], forward_world[..., 0])
    ).wxyz
    assert R_canonical_world.shape == (B, T, 4)
    cond_parts.append(
        ru.matrix_to_rotation_6d((SO3(R_canonical_world) @ SO3(R_world_cpf)).as_matrix()),
    )

    head = torch.cat(cond_parts, dim=-1) # (B, T, 16)
    #####################################################################################################################################
    global_orient = ru.quaternion_to_matrix(batch.T_world_root[..., :4]).unsqueeze(2) # [B, T, 1, 3, 3]
    body_pose = ru.quaternion_to_matrix(batch.body_quats) # [B, T, 21, 3, 3]
    root_trans = batch.T_world_root[..., 4:] # [B, T, 3]
    
    local_rotation_body = ru.matrix_to_rotation_6d(torch.cat((global_orient, body_pose), dim=2)).reshape(B, T, -1) # [B, T, 132]
    motion = torch.cat((local_rotation_body, root_trans), dim=-1) # [B, T, 135]
    #####################################################################################################################################

    return head, motion

def input_data_egoallo_plus1(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EgoAllo-style input construction when the dataloader returns subseq_len + 1 frames.

    Assumptions:
      batch.T_world_cpf  : (B, Tfull, 7)  wxyz_xyz
      batch.T_world_root : (B, Tfull, 7)  wxyz_xyz
      batch.body_quats   : (B, Tfull, 21, 4)  wxyz (root excluded)
      (optional) batch.hand_quats : (B, Tfull, 30, 4) or None

    Output:
      head   : (B, T, 16) where T = Tfull - 1
               [delta_R6, delta_t3, height1, yaw_R6]
      motion : (B, T, 135)
               [ (root_R6 + 21*R6)=132 , root_trans3 ]
    """
    # -------------------------
    # Shapes / time alignment
    # -------------------------
    B, Tfull, _ = batch.T_world_cpf.shape
    assert Tfull >= 2, f"Need >=2 frames, got {Tfull}"
    T = Tfull - 1

    # print(T)
    # -------------------------
    # (A) Relative CPF transform: T_cpf_tm1_cpf_t for t=1..Tfull-1
    #     length T
    # -------------------------
    T_cpf_tm1_cpf_t = SE3(batch.T_world_cpf[:, :-1, :]).inverse() @ SE3(batch.T_world_cpf[:, 1:, :])
    T_cpf_tm1_cpf_t = T_cpf_tm1_cpf_t.as_matrix()  # (B, T, 4,4) or (B, T, 3,4)

    # Always safe:
    T_cpf_tm1_cpf_t_R = T_cpf_tm1_cpf_t[..., :3, :3]   # (B, T, 3,3)
    T_cpf_tm1_cpf_t_P = T_cpf_tm1_cpf_t[..., :3, 3]    # (B, T, 3)

    delta_R6 = ru.matrix_to_rotation_6d(T_cpf_tm1_cpf_t_R)   # (B, T, 6)
    delta_t3 = T_cpf_tm1_cpf_t_P                              # (B, T, 3)

    # -------------------------
    # (B) Height from floor for frames 1..Tfull-1 (length T)
    # -------------------------
    height_from_floor = batch.T_world_cpf[:, 1:, 6:7]  # (B, T, 1)

    # -------------------------
    # (C) Canonical yaw conditioning for frames 1..Tfull-1 (length T)
    # -------------------------
    R_world_cpf_quat = SE3(batch.T_world_cpf[:, 1:, :]).rotation().wxyz  # (B, T, 4)
    forward_cpf = R_world_cpf_quat.new_tensor([0.0, 0.0, 1.0])           # (3,)
    forward_world = SO3(R_world_cpf_quat) @ forward_cpf                  # (B, T, 3)

    yaw = -torch.arctan2(forward_world[..., 1], forward_world[..., 0])   # (B, T)
    R_canonical_world = SO3.from_z_radians(yaw).wxyz                     # (B, T, 4)

    yaw_R6 = ru.matrix_to_rotation_6d(
        (SO3(R_canonical_world) @ SO3(R_world_cpf_quat)).as_matrix()
    )  # (B, T, 6)

    # head: (B, T, 16) = 6 + 3 + 1 + 6
    head = torch.cat([delta_R6, delta_t3, height_from_floor, yaw_R6], dim=-1)

    # -------------------------
    # (D) Motion input for frames 1..Tfull-1 (length T)
    #     root orient from T_world_root, body pose from body_quats, plus root translation
    # -------------------------
    root_R = ru.quaternion_to_matrix(batch.T_world_root[:, 1:, :4]).unsqueeze(2)  # (B, T, 1, 3,3)
    body_R = ru.quaternion_to_matrix(batch.body_quats[:, 1:, ...])                # (B, T, 21,3,3)
    root_t = batch.T_world_root[:, 1:, 4:]                                        # (B, T, 3)

    # (B, T, 22, 3,3) -> rotation6d -> (B, T, 22, 6) -> flatten (B, T, 132)
    motion_rot6 = ru.matrix_to_rotation_6d(torch.cat([root_R, body_R], dim=2)).reshape(B, T, -1)
    motion = torch.cat([motion_rot6, root_t], dim=-1)  # (B, T, 135)

    # -------------------------
    # Final sanity checks
    # -------------------------
    assert head.shape == (B, T, 16), f"head shape {head.shape} != {(B, T, 16)}"
    assert motion.shape == (B, T, 135), f"motion shape {motion.shape} != {(B, T, 135)}"
    return head, motion

    
def input_data_inference(T_world_cpf):
    #####################################################################################################################################
    T_cpf_tm1_cpf=(SE3(T_world_cpf[:, :-1, :]).inverse() @ SE3(T_world_cpf[:, 1:, :])).parameters()
    
    T_cpf_tm1_cpf_t = SE3(T_cpf_tm1_cpf).as_matrix()[..., :3, :]
    T_cpf_tm1_cpf_t_R = ru.matrix_to_rotation_6d(T_cpf_tm1_cpf_t[..., :3]) # (B, T, 6)
    T_cpf_tm1_cpf_t_P = T_cpf_tm1_cpf_t[..., 3:].squeeze(-1) # (B, T, 3)
    #####################################################################################################################################
    T_world_cpf = T_world_cpf[:, 1:, :]
    B, T, _ = T_world_cpf.shape
    #####################################################################################################################################
    height_from_floor = T_world_cpf[..., 6:7]
    #####################################################################################################################################
    R_world_cpf = SE3(T_world_cpf).rotation().wxyz
    forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
    forward_world = SO3(R_world_cpf) @ forward_cpf
    assert forward_world.shape == (B, T, 3)
    R_canonical_world = SO3.from_z_radians(
        -torch.arctan2(forward_world[..., 1], forward_world[..., 0])
    ).wxyz
    assert R_canonical_world.shape == (B, T, 4)
    #####################################################################################################################################
    cond_parts = [
        T_cpf_tm1_cpf_t_R,
        T_cpf_tm1_cpf_t_P,
        height_from_floor,
        ru.matrix_to_rotation_6d((SO3(R_canonical_world) @ SO3(R_world_cpf)).as_matrix()),
    ]    
    head = torch.cat(cond_parts, dim=-1) # (B, T, 16)
    return head


def input_data_all(batch):
    B, T, _ = batch.T_world_cpf.shape
    # print(batch.T_world_cpf.shape)

    #####################################################################################################################################
    height_from_floor = batch.T_world_cpf[..., 6:7]
    #####################################################################################################################################
    T_cpf_tm1_cpf_t = SE3(batch.T_cpf_tm1_cpf_t).as_matrix()[..., :3, :]
    T_cpf_tm1_cpf_t_R = ru.matrix_to_rotation_6d(T_cpf_tm1_cpf_t[..., :3]) # (B, T, 6)
    T_cpf_tm1_cpf_t_P = T_cpf_tm1_cpf_t[..., 3:].squeeze() # (B, T, 3)
    
    cond_parts = [
        T_cpf_tm1_cpf_t_R,
        T_cpf_tm1_cpf_t_P,
        height_from_floor,
    ]

    R_world_cpf = SE3(batch.T_world_cpf).rotation().wxyz
    forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
    forward_world = SO3(R_world_cpf) @ forward_cpf
    assert forward_world.shape == (B, T, 3)
    R_canonical_world = SO3.from_z_radians(
        -torch.arctan2(forward_world[..., 1], forward_world[..., 0])
    ).wxyz
    assert R_canonical_world.shape == (B, T, 4)
    cond_parts.append(
        ru.matrix_to_rotation_6d((SO3(R_canonical_world) @ SO3(R_world_cpf)).as_matrix()),
    )

    head = torch.cat(cond_parts, dim=-1) # (B, T, 16)
    #####################################################################################################################################
    global_orient = ru.quaternion_to_matrix(batch.T_world_root[..., :4]).unsqueeze(2) # [B, T, 1, 3, 3]
    body_pose = ru.quaternion_to_matrix(batch.body_quats) # [B, T, 21, 3, 3]
    root_trans = batch.T_world_root[..., 4:] # [B, T, 3]
    
    local_rotation_body = ru.matrix_to_rotation_6d(torch.cat((global_orient, body_pose), dim=2)).reshape(B, T, -1) # [B, T, 132]
    motion = torch.cat((local_rotation_body, root_trans,), dim=-1) # [B, T, 135]
    #####################################################################################################################################

    return head, motion    
    
def input_data_all_2(batch):
    B, T, _ = batch.T_world_cpf.shape
    # print(batch.T_world_cpf.shape)

    #####################################################################################################################################
    height_from_floor = batch.T_world_cpf[..., 6:7]
    #####################################################################################################################################
    T_cpf_tm1_cpf_t = SE3(batch.T_cpf_tm1_cpf_t).as_matrix()[..., :3, :]
    T_cpf_tm1_cpf_t_R = ru.matrix_to_rotation_6d(T_cpf_tm1_cpf_t[..., :3]) # (B, T, 6)
    T_cpf_tm1_cpf_t_P = T_cpf_tm1_cpf_t[..., 3:].squeeze() # (B, T, 3)
    
    cond_parts = [
        T_cpf_tm1_cpf_t_R,
        T_cpf_tm1_cpf_t_P,
        height_from_floor,
    ]

    R_world_cpf = SE3(batch.T_world_cpf).rotation().wxyz
    forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
    forward_world = SO3(R_world_cpf) @ forward_cpf
    assert forward_world.shape == (B, T, 3)
    R_canonical_world = SO3.from_z_radians(
        -torch.arctan2(forward_world[..., 1], forward_world[..., 0])
    ).wxyz
    assert R_canonical_world.shape == (B, T, 4)
    cond_parts.append(
        ru.matrix_to_rotation_6d((SO3(R_canonical_world) @ SO3(R_world_cpf)).as_matrix()),
    )

    head = torch.cat(cond_parts, dim=-1) # (B, T, 16)
    #####################################################################################################################################
    global_orient = ru.quaternion_to_matrix(batch.T_world_root[..., :4]).unsqueeze(2) # [B, T, 1, 3, 3]
    body_pose = ru.quaternion_to_matrix(batch.body_quats) # [B, T, 21, 3, 3]
    root_trans = batch.T_world_root[..., 4:] # [B, T, 3]
    
    local_rotation_body = ru.matrix_to_rotation_6d(torch.cat((global_orient, body_pose), dim=2)).reshape(B, T, -1) # [B, T, 132 = 6+126]

    motion = torch.cat((local_rotation_body, root_trans), dim=-1) # [B, T, 132+3]
    #####################################################################################################################################

    return head, motion   

def input_data_all_2_plus1(batch):
    """
    EgoAllo-style input construction when the dataloader returns subseq_len + 1 frames.

    Assumptions:
      batch.T_world_cpf  : (B, Tfull, 7)  wxyz_xyz
      batch.T_world_root : (B, Tfull, 7)  wxyz_xyz
      batch.body_quats   : (B, Tfull, 21, 4)  wxyz (root excluded)
      (optional) batch.hand_quats : (B, Tfull, 30, 4) or None

    Output:
      head   : (B, T, 16) where T = Tfull - 1
               [delta_R6, delta_t3, height1, yaw_R6]
      motion : (B, T, 135)
               [ (root_R6 + 21*R6)=132 , root_trans3 ]
    """
    # -------------------------
    # Shapes / time alignment
    # -------------------------
    B, Tfull, _ = batch.T_world_cpf.shape
    assert Tfull >= 2, f"Need >=2 frames, got {Tfull}"
    T = Tfull - 1

    # -------------------------
    # (A) Relative CPF transform: T_cpf_tm1_cpf_t for t=1..Tfull-1
    #     length T
    # -------------------------
    T_cpf_tm1_cpf_t = SE3(batch.T_world_cpf[:, :-1, :]).inverse() @ SE3(batch.T_world_cpf[:, 1:, :])
    T_cpf_tm1_cpf_t = T_cpf_tm1_cpf_t.as_matrix()  # (B, T, 4,4) or (B, T, 3,4)

    # Always safe:
    T_cpf_tm1_cpf_t_R = T_cpf_tm1_cpf_t[..., :3, :3]   # (B, T, 3,3)
    T_cpf_tm1_cpf_t_P = T_cpf_tm1_cpf_t[..., :3, 3]    # (B, T, 3)

    T_cpf_tm1_cpf_t_R = ru.matrix_to_rotation_6d(T_cpf_tm1_cpf_t_R)   # (B, T, 6)
    T_cpf_tm1_cpf_t_P = T_cpf_tm1_cpf_t_P                              # (B, T, 3)

    # -------------------------
    # (B) Height from floor for frames 1..Tfull-1 (length T)
    # -------------------------
    height_from_floor = batch.T_world_cpf[:, 1:, 6:7]  # (B, T, 1)

    cond_parts = [
        T_cpf_tm1_cpf_t_R,
        T_cpf_tm1_cpf_t_P,
        height_from_floor,
    ] # (B, T, 10)

    R_world_cpf = SE3(batch.T_world_cpf[:, 1:, :]).rotation().wxyz
    forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
    forward_world = SO3(R_world_cpf) @ forward_cpf
    assert forward_world.shape == (B, T, 3)
    R_canonical_world = SO3.from_z_radians(
        -torch.arctan2(forward_world[..., 1], forward_world[..., 0])
    ).wxyz
    assert R_canonical_world.shape == (B, T, 4)
    cond_parts.append(
        ru.matrix_to_rotation_6d((SO3(R_canonical_world) @ SO3(R_world_cpf)).as_matrix()),
    ) # (B, T, 10 + 6)

    head = torch.cat(cond_parts, dim=-1) # (B, T, 16)
    
    #####################################################################################################################################
    global_orient = ru.quaternion_to_matrix(batch.T_world_root[:, 1:, :4]).unsqueeze(2) # [B, T, 1, 3, 3]
    body_pose = ru.quaternion_to_matrix(batch.body_quats[:, 1:, ...]) # [B, T, 21, 3, 3]
    root_trans = batch.T_world_root[:, 1:, 4:] # [B, T, 3]
    
    local_rotation_body = ru.matrix_to_rotation_6d(torch.cat((global_orient, body_pose), dim=2)).reshape(B, T, -1) # [B, T, 132 = 6+126]

    motion = torch.cat((local_rotation_body, root_trans), dim=-1) # [B, T, 135]
    #####################################################################################################################################

    return head, motion   

def input_data_new_train(batch):
    B, T, _ = batch.T_world_cpf.shape
    # print(batch.T_world_cpf.shape)

    #####################################################################################################################################
    height_from_floor = batch.T_world_cpf[..., 6:7]
    #####################################################################################################################################
    T_cpf_tm1_cpf_t = SE3(batch.T_cpf_tm1_cpf_t).as_matrix()[..., :3, :]
    T_cpf_tm1_cpf_t_R = ru.matrix_to_rotation_6d(T_cpf_tm1_cpf_t[..., :3]) # (B, T, 6)
    T_cpf_tm1_cpf_t_P = T_cpf_tm1_cpf_t[..., 3:].squeeze(-1) # (B, T, 3)
    
    cond_parts = [
        T_cpf_tm1_cpf_t_R,
        T_cpf_tm1_cpf_t_P,
        height_from_floor,
    ]

    R_world_cpf = SE3(batch.T_world_cpf).rotation().wxyz
    forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
    forward_world = SO3(R_world_cpf) @ forward_cpf
    assert forward_world.shape == (B, T, 3)
    R_canonical_world = SO3.from_z_radians(
        -torch.arctan2(forward_world[..., 1], forward_world[..., 0])
    ).wxyz
    assert R_canonical_world.shape == (B, T, 4)
    cond_parts.append(
        ru.matrix_to_rotation_6d((SO3(R_canonical_world) @ SO3(R_world_cpf)).as_matrix()),
    )

    head = torch.cat(cond_parts, dim=-1) # (B, T, 16)
    #####################################################################################################################################
    # global_orient = ru.quaternion_to_matrix(batch.T_world_root[..., :4]).unsqueeze(2) # [B, T, 1, 3, 3]
    body_pose = ru.quaternion_to_matrix(batch.body_quats) # [B, T, 21, 3, 3]
    # root_trans = batch.T_world_root[..., 4:] # [B, T, 3]
    
    betas = batch.betas # [B, T, 16]
    contacts = batch.contacts # [B, T, 21]
    
    local_rotation_body = ru.matrix_to_rotation_6d(body_pose).reshape(B, T, -1) # [B, T, 126]

    motion = torch.cat((local_rotation_body, betas, contacts), dim=-1) # [B, T, 126 + 16 + 21]
    # print(motion.shape)
    #####################################################################################################################################

    return head, motion    
    
    
# def extract_joint_vertices(outputs, batch, smplh, flag_joints=True, flag_vertices=True):
#     B, T, _ = batch.T_world_cpf.shape
    
#     # Pred ---------------------------------------------------------------------------------------------------
#     pose_from_head = ru.matrix_to_axis_angle(ru.rotation_6d_to_matrix(outputs['recon_from_head'].reshape(B, T, 22, 6))).reshape(B*T, 66) # (B, T, 66)
#     pose_from_motion = ru.matrix_to_axis_angle(ru.rotation_6d_to_matrix(outputs['recon_from_motion'].reshape(B, T, 22, 6))).reshape(B*T, 66) # (B, T, 66)
    
#     if flag_joints:
#         head_smpl_input_params = {
#                 'global_orient': pose_from_head[..., :3],
#                 'body_pose': pose_from_head[..., 3:],
#                 'betas': batch.betas.reshape(B*T, -1)[:, :10],
#                 'transl': torch.zeros_like(batch.T_world_root[..., 4:].reshape(B*T, -1)),
#             }
#         head_smpl_output = smplh(**{k: v for k,v in head_smpl_input_params.items()}, pose2rot=True)
        
#         if flag_vertices:
#             motion_smpl_input_params = {
#                     'global_orient': pose_from_motion[..., :3],
#                     'body_pose': pose_from_motion[..., 3:],
#                     'betas': batch.betas.reshape(B*T, -1)[:, :10],
#                     'transl': torch.zeros_like(batch.T_world_root[..., 4:].reshape(B*T, -1)),
                    
#                 }
#             motion_smpl_output = smplh(**{k: v for k,v in motion_smpl_input_params.items()}, pose2rot=True)
#     # --------------------------------------------------------------------------------------------------------
    
#     # GT   ---------------------------------------------------------------------------------------------------
#     if flag_vertices:
#         with torch.no_grad(): 
            
#             gt_global_orient_aa = ru.quaternion_to_axis_angle(batch.T_world_root[..., :4].reshape(B * T, 4)).squeeze(1)
#             gt_body_pose_aa = ru.quaternion_to_axis_angle(batch.body_quats.reshape(B * T, -1, 4)).reshape(B * T, -1)
#             gt_betas = batch.betas.reshape(B*T, -1)[:, :10]
#             gt_smpl_input_params = {
#                 'global_orient': gt_global_orient_aa,
#                 'body_pose': gt_body_pose_aa,
#                 'betas': gt_betas,
#                 'transl': (batch.T_world_root[..., 4:]- batch.T_world_root[:, [0], 4:]).reshape(B*T, -1),
#             }

#             gt_smpl_output = smplh(**{k: v for k,v in gt_smpl_input_params.items()}, pose2rot=True)
#     # --------------------------------------------------------------------------------------------------------
    
#     head_keypoints_3d = motion_keypoints_3d = gt_keypoints_3d = None
#     head_vertices = motion_vertices = gt_vertices = None
    
#     if flag_joints:
#         head_keypoints_3d = head_smpl_output.joints[:, :22, :].reshape(B, T, 22, 3)    # (B, T, 22, 3)
#         if flag_vertices:
#             motion_keypoints_3d = motion_smpl_output.joints[:, :22, :].reshape(B, T, 22, 3)    # (B, T, 22, 3)
#         gt_keypoints_3d = torch.cat((batch.T_world_root[..., 4:].unsqueeze(-2), batch.joints_wrt_world), dim=-2) # (B, T, 22, 3)
    
#     if flag_vertices:
#         gt_keypoints_3d = gt_smpl_output.joints[:, :22, :].reshape(B, T, 22, 3)    # (B, T, 22, 3)
        
#         head_vertices = head_smpl_output.vertices.reshape(B, T, 6890, 3)                 # (B, T, 6890, 3)
#         motion_vertices = motion_smpl_output.vertices.reshape(B, T, 6890, 3)                 # (B, T, 6890, 3)
#         gt_vertices = gt_smpl_output.vertices.reshape(B, T, -1, 3)                # (B, T, 6890, 3)
        
#     out_dict = {
#         'head_keypoints_3d': head_keypoints_3d,
#         'head_vertices': head_vertices,
#         'motion_keypoints_3d': motion_keypoints_3d,
#         'motion_vertices': motion_vertices,
#         'gt_keypoints_3d': gt_keypoints_3d,
#         'gt_vertices': gt_vertices,
#     }

#     return out_dict



def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
import csv

def save_loss_to_csv(csv_file_name, loss_dict):

    fieldnames = loss_dict.keys()
    processed_dict = {key: value.item() if isinstance(value, torch.Tensor) else value for key, value in loss_dict.items()}
    try:
        with open(csv_file_name, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            if list(fieldnames) != header:
                raise ValueError("CSV 파일의 헤더가 딕셔너리의 키와 다릅니다.")
            
    except FileNotFoundError:
        with open(csv_file_name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_file_name, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(processed_dict)

    

    
    