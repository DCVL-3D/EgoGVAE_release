import os
import torch
import torch.utils.data
import numpy as np

from pathlib import Path
from tqdm import tqdm
from smplx import SMPLH

# Data
from data.amass import AmassHdf5Dataset
from data.dataclass import collate_dataclass
# from data.body_model import BodyModel

# Network
from model.network_3 import Head2Motion

# Utils
from model.utils import input_data, set_seed, input_data_egoallo, input_data_all_2_plus1
from model.utils_checkpoint import load_checkpoint
from utils.smplh_egoallo import extract_joints_vertices
from utils.smplh_henu import extract_joints_vertices_henu

# Visualization
# from utils.vis_mesh import visualize_result_png
from utils.vis_z import visualize_z, visualize_from_dist_lists, visualize_from_dist_lists_with_prior, visualize_tsne_rsample_dense, visualize_tsne_rsample_dense_aligned_to_ref, plot_tsne_rsample_with_axis_file


# Metric
from model.metric import  print_metrics, egoallo_metric, egoallo_metric_paper, metric_for_cvpr, print_metrics_for_cvpr, metric_for_cvpr_post_guided

# Config
from config.config_3 import *

import time

import torch

# EgoAllo 함수(원본) 사용
# from utils.guidance_optimization import do_guidance_optimization
from utils.from_egoallo import fncsmpl

import torch
import numpy as np
from quat import *

def forward_test_noncausal_first_then_slide(
    model,
    head: torch.Tensor,     # (B, T, Dh)
    motion: torch.Tensor,   # (B, T, Dm)
    window_size: int = 128,
    time_keys=("head_betas", "head_contacts", "head_theta", "pose_quat"),
    z_keys=("head_z", "motion_z"),
):
    """
    non-causal long-seq inference:
      - first window [0:W] => fill frames 0..W-1 with all timesteps
      - for t >= W, use sliding window [t-W+1 : t+1] and only take last timestep output

    Returns:
      dict of (B, T, ...) for time_keys and z_keys (z is broadcast/blended as described below)
      plus dist entries as-is (optional)
    """
    assert head.ndim == 3 and motion.ndim == 3
    B, T, Dh = head.shape
    _, Tm, Dm = motion.shape
    assert T == Tm, f"T mismatch: head T={T}, motion T={Tm}"

    device = head.device
    W = window_size
    assert T >= 1

    # ---- allocate outputs
    final = {}

    # we need one forward to know shapes for allocation
    if T >= W:
        head0 = head[:, :W, :]
        mot0 = motion[:, :W, :]
    else:
        # T < W면 그냥 T만으로 한 번에 처리 (패딩/리사이즈는 모델 정의에 따름)
        head0 = head
        mot0 = motion

    with torch.inference_mode():
        out0 = model.forward_test(head0, mot0)

    # allocate time outputs
    for k in time_keys:
        # out0[k]: (B, W0, ...)
        v0 = out0[k]
        # target: (B, T, ...)
        final[k] = torch.zeros((B, T, *v0.shape[2:]), device=device, dtype=v0.dtype)

    # allocate z outputs (model이 z를 (B, latent)로 주는 전제)
    for k in z_keys:
        z0 = out0[k]
        final[k] = torch.zeros((B, T, z0.shape[-1]), device=device, dtype=z0.dtype)

    # keep dist (원하면 frame별로 확장 가능)
    final["head_dist"] = out0.get("head_dist", None)
    final["motion_dist"] = out0.get("motion_dist", None)

    # ---- case: short sequence
    if T <= W:
        with torch.inference_mode():
            out = model.forward_test(head, motion)
        # time: 그대로 채움
        for k in time_keys:
            final[k][:, :T, ...] = out[k][:, :T, ...]
        # z: 모든 프레임에 동일하게 broadcast
        for k in z_keys:
            final[k][:, :T, :] = out[k].unsqueeze(1).expand(-1, T, -1)
        final["head_dist"] = out.get("head_dist", final["head_dist"])
        final["motion_dist"] = out.get("motion_dist", final["motion_dist"])
        return final

    # ---- 1) first window: fill 0..W-1 using all timesteps
    with torch.inference_mode():
        out_first = out0  # already computed

    for k in time_keys:
        final[k][:, :W, ...] = out_first[k]  # (B,W,...)

    for k in z_keys:
        z = out_first[k]  # (B,latent)
        final[k][:, :W, :] = z.unsqueeze(1).expand(-1, W, -1)

    # ---- 2) sliding windows for t = W .. T-1, take only last timestep
    # build indices for all sliding windows at once: (Nwin, W)
    # window ending at t => [t-W+1 ... t]
    t_ends = torch.arange(W, T, device=device)  # t=W..T-1
    Nwin = t_ends.numel()
    base = torch.arange(W, device=device)  # 0..W-1
    idx = (t_ends[:, None] - (W - 1 - base[None, :]))  # (Nwin, W)

    # gather: (B, Nwin, W, D) -> (B*Nwin, W, D)
    head_in = head[:, idx, :].reshape(B * Nwin, W, Dh)
    mot_in  = motion[:, idx, :].reshape(B * Nwin, W, Dm)

    with torch.inference_mode():
        out = model.forward_test(head_in, mot_in)

    # time: last timestep only -> frame t
    for k in time_keys:
        v_last = out[k][:, -1, ...]  # (B*Nwin, ...)
        v_last = v_last.reshape(B, Nwin, *v_last.shape[1:])
        # write to frames [W..T-1]
        final[k][:, W:T, ...] = v_last

    # z: (B*Nwin, latent) -> (B, Nwin, latent) -> write to frames [W..T-1]
    for k in z_keys:
        z = out[k].reshape(B, Nwin, -1)
        final[k][:, W:T, :] = z

    # dist는 여기서는 마지막 호출의 dist로 덮어씀(원하면 리스트로 저장)
    final["head_dist"] = out.get("head_dist", final["head_dist"])
    final["motion_dist"] = out.get("motion_dist", final["motion_dist"])

    return final

def _canonical_overlap_weights(seq_len: int, overlap_size: int, device, dtype=torch.float32):
    """
    EgoAllo snippet과 동일한 형태의 canonical weights를 seq_len 길이로 생성.
    (우리는 window마다 [:win_len]만 잘라서 씀)
    """
    if overlap_size <= 0:
        return torch.ones((seq_len,), device=device, dtype=dtype)

    w = (
        torch.from_numpy(
            np.minimum(
                overlap_size,
                np.minimum(
                    np.arange(1, seq_len + 1),
                    np.arange(1, seq_len + 1)[::-1],
                ),
            ).astype(np.float32)
            / float(overlap_size)
        )
        .to(device=device, dtype=dtype)
    )
    return w  # (seq_len,)


def forward_test_noncausal_overlap_egoallo_style(
    model,
    head: torch.Tensor,     # (B, T, Dh)
    motion: torch.Tensor,   # (B, T, Dm)
    window_size: int = 128,
    overlap_size: int = 32,
    time_keys=("head_betas", "head_contacts", "head_theta", "pose_quat"),
    z_keys=("head_z", "motion_z"),
    eps: float = 1e-8,
):
    """
    EgoAllo-style overlap blending for long sequences:
      - windows: start = 0, stride, 2*stride, ...
      - end = min(start+W, T)  (마지막 window는 짧을 수 있음)
      - overlap_weights_slice = canonical[:win_len]
      - pred += out * w
      - weights += w
      - pred /= weights

    Returns:
      dict:
        time_keys -> (B, T, ...)
        z_keys    -> (B, T, latent)  (window별 z를 per-timestep로 펼쳐서 동일하게 블렌딩)
        head_dist/motion_dist -> 마지막 window의 dist (원하면 리스트로 바꾸면 됨)
    """
    assert head.ndim == 3 and motion.ndim == 3
    B, T, Dh = head.shape
    _, Tm, Dm = motion.shape
    assert T == Tm, f"T mismatch: head T={T}, motion T={Tm}"

    device = head.device
    W = window_size
    O = overlap_size
    stride = W - O
    assert stride > 0, "overlap_size must be < window_size"

    # EgoAllo처럼 "전체 길이 T"에 대한 canonical을 만들고, window마다 앞에서 잘라 씀
    canonical_w = _canonical_overlap_weights(T, O, device=device, dtype=torch.float32)

    # 1) 첫 window 한 번 돌려서 output shape 확보
    s0, e0 = 0, min(W, T)
    with torch.inference_mode():
        out0 = model.forward_test(head[:, s0:e0, :], motion[:, s0:e0, :])

    final = {}

    # 2) 누적 버퍼 + weight 버퍼 생성
    for k in time_keys:
        v0 = out0[k]  # (B, win_len, ...)
        final[k] = torch.zeros((B, T, *v0.shape[2:]), device=device, dtype=v0.dtype)
        final[f"_{k}_w"] = torch.zeros((1, T, 1), device=device, dtype=torch.float32)

    for k in z_keys:
        z0 = out0[k]  # (B, latent)
        final[k] = torch.zeros((B, T, z0.shape[-1]), device=device, dtype=z0.dtype)
        final[f"_{k}_w"] = torch.zeros((1, T, 1), device=device, dtype=torch.float32)

    final["head_dist"] = out0.get("head_dist", None)
    final["motion_dist"] = out0.get("motion_dist", None)

    # 3) window loop (EgoAllo: range(0, T, stride))
    with torch.inference_mode():
        for start_t in range(0, T, stride):
            end_t = min(start_t + W, T)
            win_len = end_t - start_t
            assert win_len > 0

            # weight slice: (1, win_len, 1)
            w_slice = canonical_w[None, :win_len, None]  # EgoAllo와 동일하게 앞쪽에서 잘라씀

            out = model.forward_test(
                head[:, start_t:end_t, :],
                motion[:, start_t:end_t, :],
            )

            # time keys: (B, win_len, ...) * w 누적
            for k in time_keys:
                pred = out[k]
                w_b = w_slice
                # pred.ndim = 3 + extra -> w를 broadcast
                for _ in range(pred.ndim - 3):
                    w_b = w_b.unsqueeze(-1)  # (1, win_len, 1, 1, ...)
                final[k][:, start_t:end_t, ...] += pred * w_b
                final[f"_{k}_w"][:, start_t:end_t, :] += w_slice

            # z keys: (B, latent) -> (B, win_len, latent)로 펼쳐서 동일 누적
            for k in z_keys:
                z = out[k]  # (B, latent)
                z_t = z.unsqueeze(1).expand(-1, win_len, -1)  # (B, win_len, latent)
                final[k][:, start_t:end_t, :] += z_t * w_slice
                final[f"_{k}_w"][:, start_t:end_t, :] += w_slice

            final["head_dist"] = out.get("head_dist", final["head_dist"])
            final["motion_dist"] = out.get("motion_dist", final["motion_dist"])

    # 4) 평균 (EgoAllo: pred /= overlap_weights)
    for k in time_keys:
        wsum = final[f"_{k}_w"]  # (1, T, 1)
        denom = wsum
        for _ in range(final[k].ndim - 3):
            denom = denom.unsqueeze(-1)
        final[k] = final[k] / (denom + eps)
        del final[f"_{k}_w"]

    for k in z_keys:
        wsum = final[f"_{k}_w"]  # (1, T, 1)
        final[k] = final[k] / (wsum + eps)
        del final[f"_{k}_w"]

    return final


def forward_test_pad_firstobs_eval(model, head, motion, window_size: int):
    """
    head:   (B0, L, Dh)
    motion: (B0, L, Dm)
    return: forward_test()의 time-key들은 (B0, L, ...)로 복원해서 반환
            (모델 입력은 (B0*L, window_size, D)로 들어감)
    """
    B0, L, Dh = head.shape
    _, Lm, Dm = motion.shape
    assert L == Lm, f"L mismatch: head L={L}, motion L={Lm}"
    device = head.device
    W = window_size

    # idx: (L, W)  each row t = [t-W+1 ... t], clamped to 0 => x0 padding
    t_idx = torch.arange(L, device=device)[:, None]      # (L,1)
    w_idx = torch.arange(W, device=device)[None, :]      # (1,W)
    idx = (t_idx - (W - 1 - w_idx)).clamp(min=0)         # (L,W)

    # windows: (B0, L, W, D) -> reshape to (B0*L, W, D)
    head_in = head[:, idx, :].reshape(B0 * L, W, Dh)
    motion_in = motion[:, idx, :].reshape(B0 * L, W, Dm)

    out = model.forward_test(head_in, motion_in)  # time outputs: (B0*L, W, ...)

    # 네 forward_test 기준 key들
    time_keys = ["head_betas", "head_contacts", "head_theta", "pose_quat"]
    z_keys = ["head_z", "motion_z"]

    final = {}

    # time outputs: 마지막 timestep만 사용 -> (B0*L, ...) -> (B0, L, ...)
    for k in time_keys:
        v_last = out[k][:, -1, ...]  # (B0*L, ...)
        final[k] = v_last.reshape(B0, L, *v_last.shape[1:])

    # z: (B0*L, latent) -> (B0, L, latent)
    for k in z_keys:
        v = out[k]
        final[k] = v.reshape(B0, L, -1)

    # dist는 보통 frame별로 들고가지 않음(원하면 바꿀 수 있음)
    final["head_dist"] = out["head_dist"]
    final["motion_dist"] = out["motion_dist"]

    return final


def eval():
    HDF5_PATH = DataConfig.HDF5_PATH
    FILE_LIST_PATH = DataConfig.FILE_LIST_PATH
    SUBSEQ_LEN = DataConfig.SUBSEQ_LEN
    
    RESULT_IMG_PATH = EvalConfig.RESULT_IMG_PATH
    RESULT_Z_PATH = EvalConfig.RESULT_Z_PATH
    CHECKPOINT_PATH = EvalConfig.CHECKPOINT_PATH
    BATCH_SIZE = EvalConfig.BATCH_SIZE
    NUM_WORKERS = EvalConfig.NUM_WORKERS

    VISUAL_OUTPUT_DIR = Path("./ours_plus1")
    VISUAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(CHECKPOINT_PATH)
    # set_seed(SEED)
    # print(f"✅ Random Seed fixed: {SEED}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = AmassHdf5Dataset(
        hdf5_path=HDF5_PATH,
        file_list_path=FILE_LIST_PATH,
        splits=("test",),
        subseq_len=SUBSEQ_LEN,
        cache_files=False,
        slice_strategy=EvalConfig.SLICE_STRATEGY,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_dataclass,
        pin_memory=True,
    )
    
    smplh = SMPLH(
        model_path=SMPLHConfig.MODEL_PATH,
        gender=SMPLHConfig.GENDER,
        num_betas=SMPLHConfig.NUM_BETAS,
        model_type=SMPLHConfig.MODEL_TYPE,
    ).to(device)
    egoallo_smpl = fncsmpl.SmplhModel.load(SMPLHConfig.EGOALLO_MODEL_PATH).to(device)
    
    model = Head2Motion(head_dim=EvalConfig.INPUT_HEAD_DIM, motion_dim=EvalConfig.INPUT_MOTION_DIM, out_dim=EvalConfig.OUPUT_MOTION_DIM).to(device)

    load_checkpoint(model, CHECKPOINT_PATH)
    
    pbar = tqdm(test_loader, desc="Evaluating")

    metrics = {
            'mpjpe': 0.0,
            'pampjpe': 0.0,
            'GND': 0.0,
            'T_head': 0.0,
            'num_samples': 0,
            
            'foot_skate_pred': 0.0,
            'foot_skate_gt': 0.0,
            'jitter_pred': 0.0,
            'jitter_gt': 0.0,
            'floating_pred': 0.0,
            'mpjve': 0.0,

    }

    z_head_list = []
    z_motion_list = []
    head_dist_list = []
    motion_dist_list = []
    # print("Sample names found in dataset._groups[:5]:", test_dataset._groups[:5])
    # print(len(test_dataset._groups))
    # print("-" * 50)    

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            # print(batch.shape)
            head, motion = input_data_egoallo(batch)
            # head, motion = input_data_all_2_plus1(batch)
            print(head.shape)
            start_time = time.time()

            # outputs = forward_test_noncausal_first_then_slide(
            #     model=model,
            #     head=head,           # (B, T, Dh)
            #     motion=motion,       # (B, T, Dm)
            #     window_size=128,
            # )

            # outputs = forward_test_noncausal_overlap_egoallo_style(
            #     model=model,
            #     head=head,           # (B, T, Dh)
            #     motion=motion,       # (B, T, Dm)
            #     window_size=128,
            #     overlap_size=32,
            # )

            # outputs = forward_test_pad_firstobs_eval(
            #     model,
            #     head=head,
            #     motion=motion,
            #     window_size=32,  # opt에서 가져오면 됨
            # )
            
            # print(head.shape)
            outputs = model.forward_test(head, motion)

            runtime = time.time() - start_time
            
            # # ============================================================
            # # Stochastic sampling from head_dist (single sample, 20 times)
            # # ============================================================
            # with torch.no_grad():
            #     dist = outputs["head_dist"]   # torch.distributions.Distribution
            #     feat_head = outputs["feat_head"]
            #     B = head.shape[0]

            #     sample_idx = 0  # 배치에서 첫 번째 샘플만 사용
            #     num_mc = 20

            #     mpjpe_list = []

            #     for s in range(num_mc):
            #         # (1) latent resampling
            #         z_sample = dist.rsample()             # (B, latent_dim)
            #         z_sample = z_sample[sample_idx:sample_idx+1]  # (1, latent_dim)

            #         feat_head = model.head_feat_dropout(feat_head)

            #         # (3) decode with resampled z
            #         head_out = model.motion_decoder(feat_head, z_sample)  # (1, T, 256)

            #         head_betas = model.head_betas_layer(head_out)
            #         head_contacts = model.head_contacts_layer(head_out)
            #         head_theta = model.head_theta_layer(head_out)

            #         outputs_mc = {
            #             # "pose_quat": pose_quat,
            #             "head_theta": head_theta,           # (B, T, 126)
            #             "head_betas": head_betas,
            #             "head_contacts": head_contacts,
            #         }
            #         # (4) metric 계산
            #         metrics_mc = metric_for_cvpr(
            #             outputs_mc,
            #             batch,
            #             egoallo_smpl
            #         )

            #         mpjpe_list.append(metrics_mc["mpjpe"])

            #     mpjpe_tensor = torch.tensor(mpjpe_list)
            #     print(
            #         f"[MC sampling | head_dist | N={num_mc}] "
            #         f"MPJPE mean: {mpjpe_tensor.mean():.4f}, "
            #         f"std: {mpjpe_tensor.std():.4f}, "
            #         f"min: {mpjpe_tensor.min():.4f}, "
            #         f"max: {mpjpe_tensor.max():.4f}"
            #     )


            # print(outputs['pose_quat'].shape)

            # # # outputs 딕셔너리와 batch 객체에서 필요한 데이터 추출
            # T_world_cpf_data = batch.T_world_cpf[0, 1:].cpu().numpy()
            # body_quats_data = outputs['pose_quat'].cpu().numpy()
            # betas_data = outputs['head_betas'].cpu().numpy() 
            # contacts_data = outputs['head_contacts'].cpu().numpy()

            # # 저장할 데이터 딕셔너리 생성
            # data_to_save = {
            #     'T_world_cpf': T_world_cpf_data,
            #     'body_quats': body_quats_data,
            #     'betas': betas_data,
            #     'contacts': contacts_data
            # }
            
            # # NPZ 파일로 저장
            # save_path = VISUAL_OUTPUT_DIR / f"sample_{i}.npz"
            # np.savez_compressed(save_path, **data_to_save)
            # # # --- 추가 코드 끝 ---

            print("RUNTIME (exclude first optimization)", runtime)
            with open("store_runtime.txt", "a") as f:
                f.write(f"{runtime}\n")
            
            # Calculate MPJPE, PAMPJPE
            metrics_batch = metric_for_cvpr(outputs, batch, egoallo_smpl)

            # metrics_batch = metric_for_cvpr_post_guided(outputs, batch, egoallo_smpl, guidance_mode="no_hands", do_guidance_optimization=do_guidance_optimization)

            print(f"\n--- Current Batch Metrics ---")
            print(f"  MPJPE: {metrics_batch['mpjpe']:.4f}")
            print(f"  PAMPJPE: {metrics_batch['pampjpe']:.4f}")
            print(f"  GND: {metrics_batch['GND']:.4f}")
            print(f"  T_head: {metrics_batch['T_head']:.4f}")

            print(f"  FS pred: {metrics_batch['foot_skate_pred']:.4f}")
            print(f"  FS gt: {metrics_batch['foot_skate_gt']:.4f}")
            print(f"  Jitter pred: {metrics_batch['jitter_pred']:.4f}")
            print(f"  Jitter gt: {metrics_batch['jitter_gt']:.4f}")
            print(f"  Floating: {metrics_batch['floating_pred']:.4f}")
            print(f"  MPJVE: {metrics_batch['mpjve']:.4f}")
            
            print(f"-----------------------------")

            metrics['mpjpe'] += metrics_batch['mpjpe']
            metrics['pampjpe'] += metrics_batch['pampjpe']
            metrics['GND'] += metrics_batch['GND']
            metrics['T_head'] += metrics_batch['T_head']
            metrics['num_samples'] += 1
            
            metrics['foot_skate_pred'] += metrics_batch['foot_skate_pred']
            metrics['foot_skate_gt'] += metrics_batch['foot_skate_gt']
            metrics['jitter_pred'] += metrics_batch['jitter_pred']
            metrics['jitter_gt'] += metrics_batch['jitter_gt']
            metrics['floating_pred'] += metrics_batch['floating_pred']
            metrics['mpjve'] += metrics_batch['mpjve']
            
            print_metrics_for_cvpr(metrics)
            
            # Visualize Z(Latent Space)
            z_head_list.append(outputs['head_z']) # z_head: (B, dim)
            z_motion_list.append(outputs['motion_z']) # z_motion: (B, dim)
            head_dist_list.append(outputs['head_dist'])
            motion_dist_list.append(outputs['motion_dist'])

            # Visualize Mesh
            # _, pred_vertices = extract_joints_vertices(metrics_batch['head_posed'], metrics_batch['head_betas'], smplh)
            # _, gt_vertices = extract_joints_vertices(metrics_batch['gt_posed'], batch.betas, smplh)
            # visualize_result_png(RESULT_IMG_PATH, pred_vertices, gt_vertices, smplh)
            
        # MPJPE, PAMPJPE
        print_metrics_for_cvpr(metrics)
        
        # Z
        z_head = torch.cat(z_head_list, dim=0) # (N*B, dim)
        z_motion = torch.cat(z_motion_list, dim=0) # (N*B, dim)

        # z_2d_10, labels_10 = visualize_tsne_rsample_dense(
        #     head_dist_list,
        #     motion_dist_list,
        #     save_path="./_TEST/tsne_epoch10.png",
        #     samples_per_dist=5,
        #     max_points_per_class=500,
        #     return_embedding=True,
        # )
        # prior_2d_10 = z_2d_10[labels_10 == 2]  # prior만 슬라이스
        # np.save("./_TEST/prior_ref_epoch10.npy", prior_2d_10)

        # visualize_tsne_rsample_dense_aligned_to_ref(
        #     head_dist_list,
        #     motion_dist_list,
        #     prior_ref_path="./_TEST/prior_ref_epoch10.npy",
        #     save_path_aligned="./_TEST/tsne_epoch200_aligned_to_10.png",
        #     samples_per_dist=5,
        #     max_points_per_class=500,
        # )



        # visualize_z(RESULT_Z_PATH, z_head=z_head, z_motion=z_motion)

        # visualize_from_dist_lists(
        #     head_dist_list,
        #     motion_dist_list,
        #     save_path="./latent_64x10.png",
        #     total_select=128,   # 전체에서 64개만 선택
        #     samples_per=10     # 각 샘플당 10회 rsample → 총 640 포인트(head+motion 합)
        # )

        # visualize_from_dist_lists_with_prior(
        #     head_dist_list,
        #     motion_dist_list,
        #     save_path="./latent_head_motion_prior.png",
        #     total_select=128,        # 전체 후보(8×44×2) 중 head+motion 합 64개만 사용
        #     samples_per=10,         # 각 샘플당 10회 rsample → head 320 + motion 320
        #     include_prior=True,     # 표준정규 포함
        #     prior_match_total=True  # prior 포인트 수를 head+motion 합과 동일하게(=640) 맞춤
        # )

        # plot_tsne_rsample_with_axis_file(
        #     head_dist_list, motion_dist_list,
        #     out_png_path="./tsne_ep_1_300.png",
        #     axis_npz_path="./tsne_axis_ref_1.npz",
        #     mode="read",
        # )
if __name__ == '__main__':
    eval()
    