import torch
import cv2
import numpy as np
import random

import model.utils_rotation as ru

from utils.from_egoallo.transforms import SE3, SO3

def input_data_all(batch):
    B, T, _ = batch.T_world_cpf.shape

    height_from_floor = batch.T_world_cpf[..., 6:7]
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

    global_orient = ru.quaternion_to_matrix(batch.T_world_root[..., :4]).unsqueeze(2) # [B, T, 1, 3, 3]
    body_pose = ru.quaternion_to_matrix(batch.body_quats) # [B, T, 21, 3, 3]

    local_rotation_body = ru.matrix_to_rotation_6d(torch.cat((global_orient, body_pose), dim=2)).reshape(B, T, -1) # [B, T, 132]
    motion = local_rotation_body

    return head, motion    

def input_data_all_egoallo(batch):
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

    local_rotation_body = ru.matrix_to_rotation_6d(torch.cat((global_orient, body_pose), dim=2)).reshape(B, T, -1) # [B, T, 132 = 6+126]

    motion = local_rotation_body

    return head, motion   

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
                raise ValueError("The CSV file header differs from the dictionary keys.")
            
    except FileNotFoundError:
        with open(csv_file_name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_file_name, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(processed_dict)

    

    
    