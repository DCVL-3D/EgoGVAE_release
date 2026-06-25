import torch
import model.utils_rotation as ru

def extract_joints_vertices_henu(outputs, batch, smplh):
    B, T, _ = batch.betas.shape
    global_orient_aa = ru.matrix_to_axis_angle(ru.rotation_6d_to_matrix(outputs['recon_from_head'][..., :6].reshape(B * T, 6))) # (B*T, 3)
    body_pose_aa = ru.matrix_to_axis_angle(ru.rotation_6d_to_matrix(outputs['recon_from_head'][..., 6:].reshape(B * T, 21, 6))).reshape(B * T, -1) # (B*T. 63)
    betas = batch.betas.reshape(B*T, -1)[:, :10]
    smpl_input_params = {
        'global_orient': global_orient_aa,
        'body_pose': body_pose_aa,
        'betas': betas,
        'transl': torch.zeros_like(batch.T_world_root[..., 4:]).reshape(B*T, -1),
    }
    
    smpl_output = smplh(**{k: v for k,v in smpl_input_params.items()}, pose2rot=True)
    pred_vertices = smpl_output.vertices.reshape(B, T, 6890, 3)  
    pred_joints = smpl_output.joints[:, 1:22, :].reshape(B, T, 21, 3)
    
    
    right_eye = (pred_vertices[..., 6260, :] + pred_vertices[..., 6262, :]) / 2.0
    left_eye = (pred_vertices[..., 2800, :] + pred_vertices[..., 2802, :]) / 2.0
    cpf_pos = (right_eye + left_eye) / 2.0
    
    align_vector = batch.T_world_cpf[..., 4:] - cpf_pos
    align_vector = align_vector - align_vector[:, [0]]
    pred_vertices = pred_vertices + align_vector.unsqueeze(-2)
    pred_joints = pred_joints + align_vector.unsqueeze(-2)
    
    
    
    
    
    global_orient_aa = ru.quaternion_to_axis_angle(batch.T_world_root[..., :4].reshape(B * T, 4)) # (B*T, 3)
    body_pose_aa = ru.quaternion_to_axis_angle(batch.body_quats.reshape(B * T, 21, 4)).reshape(B * T, -1) # (B*T. 63)
    betas = batch.betas.reshape(B*T, -1)[:, :10]
    smpl_input_params = {
        'global_orient': global_orient_aa,
        'body_pose': body_pose_aa,
        'betas': betas,
        'transl': (batch.T_world_root[..., 4:]- batch.T_world_root[:, [0], 4:]).reshape(B*T, -1),
    }

    smpl_output = smplh(**{k: v for k,v in smpl_input_params.items()}, pose2rot=True)
    gt_vertices = smpl_output.vertices.reshape(B, T, 6890, 3)  
    gt_joints = smpl_output.joints[:, 1:22, :].reshape(B, T, 21, 3)
    
    return gt_joints, gt_vertices, pred_joints, pred_vertices
