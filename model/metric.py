from utils.smplh_utils import evaluation_build_smplh_posed
from utils.from_egoallo.metrics_helpers import compute_mpjpe, compute_head_trans, compute_mpjve, compute_jitter, \
                                            compute_foot_skate_gt_contact_batched, ground_metrics_lowest

def calculate_metrics(outputs, batch, fncsmpl):

    head_posed, gt_posed, head_betas = evaluation_build_smplh_posed(outputs, batch, fncsmpl)

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

    ground = ground_metrics_lowest(
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                gt_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
            )
    
    T_head = compute_head_trans(
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
            ).mean()

    foot_skate_pred = compute_foot_skate_gt_contact_batched(
                        pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                        gt_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                        foot_indices=(6,7,9,10),
                    ).mean()

    jitter_pred = compute_jitter(
                    Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :]
                ).mean()
    
    jitter_gt = compute_jitter(
                    Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :]
                ).mean()

    mpjve = compute_mpjve(    
                label_T_world_root=gt_posed.T_world_root,
                label_Ts_world_joint=gt_posed.Ts_world_joint[:, :, :21, :],
                pred_T_world_root=head_posed.T_world_root,
                pred_Ts_world_joint=head_posed.Ts_world_joint[:, :, :21, :],
                per_frame_procrustes_align=False,
                fps = 30,
            ).mean()
    
    output = {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe,
        'GND': ground,
        'T_head': T_head,

        'foot_skate': foot_skate_pred,
        'jitter_pred': jitter_pred,
        'jitter_gt': jitter_gt,
        'mpjve': mpjve,

        'head_posed': head_posed,
        'gt_posed': gt_posed,
        'head_betas': head_betas,
    }
    
    return output


def print_metrics(metrics):
    num_samples = metrics['num_samples']
    avg_mpjpe = metrics['mpjpe'] / num_samples
    avg_pampjpe = metrics['pampjpe'] / num_samples
    avg_gnd = metrics['GND'] / num_samples
    avg_t_head = metrics['T_head'] / num_samples
    
    avg_fs_pred = metrics['foot_skate'] / num_samples
    avg_jitter_pred = metrics['jitter_pred'] / num_samples
    avg_jitter_gt = metrics['jitter_gt'] / num_samples
    avg_mpjve = metrics['mpjve'] / num_samples


    print(f"\n------------ Evaluation Results ------------")
    print(f"Average MPJPE: {avg_mpjpe:.2f} mm")
    print(f"Average PAMPJPE: {avg_pampjpe:.2f} mm")
    print(f"Average GND: {avg_gnd:.2f}")
    print(f"Average T_head: {avg_t_head:.4f} mm")

    print(f"Average FS: {avg_fs_pred:.4f} mm")
    print(f"Average Jitter pred: {avg_jitter_pred:.4f} mm")
    print(f"Average Jitter gt: {avg_jitter_gt:.4f} mm")
    print(f"Average MPJVE: {avg_mpjve:.2f} mm")
    print("----------------------------------------------")
    