import torch
import numpy as np

from utils.smplh_egoallo import build_smplh_posed, build_smplh_posed_paper, build_smplh_posed_jjh, build_smplh_posed_jjh_plus1, build_smplh_posed_adt
from utils.from_egoallo.metrics_helpers import compute_mpjpe_train, compute_mpjpe, compute_foot_skate, compute_foot_contact, compute_head_trans, compute_mpjve, compute_jitter, compute_floating_error, compute_foot_skate_gt_contact_batched, ground_metrics_lowest_egoallo
from utils.from_egoallo.metrics_helpers import procrustes_align

def egoallo_metric(outputs, batch, fncsmpl):

    pred_posed, gt_posed, pred_betas = build_smplh_posed(outputs, batch, fncsmpl)

    mpjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=pred_posed.T_world_root,
                pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
            ).mean()
    
    pampjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=pred_posed.T_world_root,
                pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=True,
            ).mean()
    

    output = {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe,
        'pred_posed': pred_posed,
        'gt_posed': gt_posed,
        'pred_betas': pred_betas
    }
    
    return output

def egoallo_metric_paper(outputs, batch, fncsmpl):

    # head_posed, motion_posed, gt_posed, head_betas, motion_betas = build_smplh_posed_paper(outputs, batch, fncsmpl)
    head_posed, gt_posed, head_betas = build_smplh_posed_jjh(outputs, batch, fncsmpl)
    
    mpjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
            ).mean()
    
    pampjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=True,
            ).mean()
    
    foot_contact = compute_foot_contact(
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    
    T_head = compute_head_trans(
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
            ).mean()
    
    # print(gt_posed.Ts_world_joint.shape)
    output = {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe,
        'GND': foot_contact,
        'T_head': T_head,
        'head_posed': head_posed,
        # 'motion_posed': motion_posed,
        'gt_posed': gt_posed,
        'head_betas': head_betas,
        # 'motion_betas': motion_betas,
        
    }
    
    return output


def egoallo_metric_train(pred_posed, gt_posed):

    mpjpe = compute_mpjpe_train(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=pred_posed.T_world_root,
                pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
            ).mean()
    
    pampjpe = compute_mpjpe_train(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=pred_posed.T_world_root,
                pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=True,
            ).mean()
    
    output = {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe,
        'pred_posed': pred_posed,
        'gt_posed': gt_posed,
    }
    
    return output


def print_metrics(metrics):
    num_samples = metrics['num_samples']
    # print(num_samples)
    avg_mpjpe = metrics['mpjpe'] / num_samples
    avg_pampjpe = metrics['pampjpe'] / num_samples
    avg_gnd = metrics['GND'] / num_samples
    avg_t_head = metrics['T_head'] / num_samples

    print(f"\n------------ Evaluation Results ------------")
    print(f"Average MPJPE: {avg_mpjpe:.2f} mm")
    print(f"Average PAMPJPE: {avg_pampjpe:.2f} mm")
    print(f"Average GND: {avg_gnd:.2f}")
    print(f"Average T_head: {avg_t_head:.4f} mm")
    print("----------------------------------------------")
    

def print_metrics_for_cvpr(metrics):
    num_samples = metrics['num_samples']
    # print(num_samples)
    avg_mpjpe = metrics['mpjpe'] / num_samples
    avg_pampjpe = metrics['pampjpe'] / num_samples
    avg_gnd = metrics['GND'] / num_samples
    avg_t_head = metrics['T_head'] / num_samples
    
    avg_fs_pred = metrics['foot_skate_pred'] / num_samples
    avg_fs_gt = metrics['foot_skate_gt'] / num_samples
    avg_jitter_pred = metrics['jitter_pred'] / num_samples
    avg_jitter_gt = metrics['jitter_gt'] / num_samples
    avg_floating = metrics['floating_pred'] / num_samples
    avg_mpjve = metrics['mpjve'] / num_samples


    print(f"\n------------ Evaluation Results ------------")
    print(f"Average MPJPE: {avg_mpjpe:.2f} mm")
    print(f"Average PAMPJPE: {avg_pampjpe:.2f} mm")
    print(f"Average GND: {avg_gnd:.2f}")
    print(f"Average T_head: {avg_t_head:.4f} mm")

    print(f"Average FS pred: {avg_fs_pred:.4f} mm")
    print(f"Average FS gt: {avg_fs_gt:.4f} mm")
    print(f"Average Jitter pred: {avg_jitter_pred:.4f} mm")
    print(f"Average Jitter gt: {avg_jitter_gt:.4f} mm")
    print(f"Average Floating: {avg_floating:.4f} mm")
    print(f"Average MPJVE: {avg_mpjve:.2f} mm")

    print("----------------------------------------------")
    



def metric_for_cvpr(outputs, batch, fncsmpl):

    # head_posed, motion_posed, gt_posed, head_betas, motion_betas = build_smplh_posed_paper(outputs, batch, fncsmpl)
    head_posed, gt_posed, head_betas = build_smplh_posed_jjh(outputs, batch, fncsmpl)
    # head_posed, gt_posed, head_betas = build_smplh_posed_jjh_plus1(outputs, batch, fncsmpl)
    
    mpjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
            ).mean()
    
    pampjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=True,
            ).mean()
    
    foot_contact = compute_foot_contact(
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    
    T_head = compute_head_trans(
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
            ).mean()
    
    # foot_skate_pred = compute_foot_skate(
    #             pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
    #         ).mean()
    
    foot_skate_pred = compute_foot_skate_gt_contact_batched(
        pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
        gt_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
        foot_indices=(6,7,9,10),
    ).mean()

    foot_skate_gt = compute_foot_skate(
                pred_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
            ).mean()
    
    jitter_pred = compute_jitter(
                Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    
    jitter_gt = compute_jitter(
                Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    
    penetration_pred, floating_pred = ground_metrics_lowest_egoallo(
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                gt_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
            )

    mpjve = compute_mpjve(    
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
                fps = 30,
            ).mean()
    

    # print(gt_posed.Ts_world_joint.shape)
    output = {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe,
        'GND': foot_contact,
        'T_head': T_head,
        'head_posed': head_posed,
        # 'motion_posed': motion_posed,
        'gt_posed': gt_posed,
        'head_betas': head_betas,
        # 'motion_betas': motion_betas,
        'foot_skate_pred': foot_skate_pred,
        'foot_skate_gt': penetration_pred,
        'jitter_pred': jitter_pred,
        'jitter_gt': jitter_gt,
        'floating_pred': floating_pred,
        'mpjve': mpjve,
    }
    
    return output


def metric_for_adt(outputs, batch, fncsmpl):

    # head_posed, motion_posed, gt_posed, head_betas, motion_betas = build_smplh_posed_paper(outputs, batch, fncsmpl)
    head_posed, gt_posed, head_betas = build_smplh_posed_adt(outputs, batch, fncsmpl)
    # head_posed, gt_posed, head_betas = build_smplh_posed_jjh_plus1(outputs, batch, fncsmpl)
    
    mpjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
            ).mean()
    
    pampjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=True,
            ).mean()
    
    foot_contact = compute_foot_contact(
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    
    T_head = compute_head_trans(
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
            ).mean()
    
    # foot_skate_pred = compute_foot_skate(
    #             pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
    #         ).mean()
    
    foot_skate_pred = compute_foot_skate_gt_contact_batched(
        pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
        gt_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
        foot_indices=(6,7,9,10),
    ).mean()

    foot_skate_gt = compute_foot_skate(
                pred_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
            ).mean()
    
    jitter_pred = compute_jitter(
                Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    
    jitter_gt = compute_jitter(
                Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    
    penetration_pred, floating_pred = ground_metrics_lowest_egoallo(
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                gt_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
            )

    mpjve = compute_mpjve(    
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
                fps = 30,
            ).mean()
    

    # print(gt_posed.Ts_world_joint.shape)
    output = {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe,
        'GND': foot_contact,
        'T_head': T_head,
        'head_posed': head_posed,
        # 'motion_posed': motion_posed,
        'gt_posed': gt_posed,
        'head_betas': head_betas,
        # 'motion_betas': motion_betas,
        'foot_skate_pred': foot_skate_pred,
        'foot_skate_gt': penetration_pred,
        'jitter_pred': jitter_pred,
        'jitter_gt': jitter_gt,
        'floating_pred': floating_pred,
        'mpjve': mpjve,
    }
    
    return output

def build_gt_posed(batch, fncsmpl):
    B, T, _ = batch.T_world_cpf.shape
    device = batch.T_world_cpf.device
    T = T - 1
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
    return gt_posed


from utils.post_guidance import build_head_posed_with_post_guidance

def metric_for_cvpr_post_guided(outputs, batch, fncsmpl, guidance_mode, do_guidance_optimization):
    # 1) head posed (post-guidance)
    head_posed, head_betas = build_head_posed_with_post_guidance(
        outputs=outputs,
        batch=batch,
        fncsmpl=fncsmpl,
        guidance_mode=guidance_mode,
        do_guidance_optimization=do_guidance_optimization,
        guidance_verbose=False,
    )

    # 2) gt posed
    gt_posed = build_gt_posed(batch, fncsmpl)

    mpjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
            ).mean()
    
    pampjpe = compute_mpjpe(
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=True,
            ).mean()
    
    foot_contact = compute_foot_contact(
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    
    T_head = compute_head_trans(
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
            ).mean()
    
    foot_skate_pred = compute_foot_skate(
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
            ).mean()

    foot_skate_gt = compute_foot_skate(
                pred_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
            ).mean()
    
    jitter_pred = compute_jitter(
                Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    
    jitter_gt = compute_jitter(
                Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :]
            ).mean()
    # print(batch.T_world_cpf.shape)
    floating_pred = ground_metrics_lowest_egoallo(
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                gt_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
            ).mean()

    mpjve = compute_mpjve(    
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
            ).mean()
    

    # print(gt_posed.Ts_world_joint.shape)
    output = {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe,
        'GND': foot_contact,
        'T_head': T_head,
        'head_posed': head_posed,
        # 'motion_posed': motion_posed,
        'gt_posed': gt_posed,
        'head_betas': head_betas,
        # 'motion_betas': motion_betas,
        'foot_skate_pred': foot_skate_pred,
        'foot_skate_gt': foot_skate_gt,
        'jitter_pred': jitter_pred,
        'jitter_gt': jitter_gt,
        'floating_pred': floating_pred,
        'mpjve': mpjve,
    }
    
    return output
