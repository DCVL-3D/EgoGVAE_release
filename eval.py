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

# Network
from model.network import Head2Motion

# Utils
from model.utils import input_data_all, input_data_all_egoallo
from model.utils_checkpoint import load_checkpoint
from utils.from_egoallo import fncsmpl
from utils.quat import *

# Metric
from model.metric import calculate_metrics, print_metrics

# Config
from config.config import *

import time

def forward_test_noncausal_first_then_slide(
    model,
    head: torch.Tensor,     # (B, T, Dh)
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
    assert head.ndim == 3
    B, T, Dh = head.shape

    device = head.device
    W = window_size
    assert T >= 1

    # ---- allocate outputs
    final = {}

    # we need one forward to know shapes for allocation
    if T >= W:
        head0 = head[:, :W, :]
    else:
        # T < W
        head0 = head

    with torch.inference_mode():
        out0 = model.forward_test(head0)

    # allocate time outputs
    for k in time_keys:
        # out0[k]: (B, W0, ...)
        v0 = out0[k]
        # target: (B, T, ...)
        final[k] = torch.zeros((B, T, *v0.shape[2:]), device=device, dtype=v0.dtype)

    for k in z_keys:
        z0 = out0[k]
        final[k] = torch.zeros((B, T, z0.shape[-1]), device=device, dtype=z0.dtype)

    final["head_dist"] = out0.get("head_dist", None)
    final["motion_dist"] = out0.get("motion_dist", None)

    # ---- case: short sequence
    if T <= W:
        with torch.inference_mode():
            out = model.forward_test(head)
        for k in time_keys:
            final[k][:, :T, ...] = out[k][:, :T, ...]
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

    with torch.inference_mode():
        out = model.forward_test(head_in)

    # time: last timestep only -> frame t
    for k in time_keys:
        v_last = out[k][:, -1, ...]  # (B*Nwin, ...)
        v_last = v_last.reshape(B, Nwin, *v_last.shape[1:])
        # write to frames [W..T-1]
        final[k][:, W:T, ...] = v_last

    for k in z_keys:
        z = out[k].reshape(B, Nwin, -1)
        final[k][:, W:T, :] = z

    final["head_dist"] = out.get("head_dist", final["head_dist"])
    final["motion_dist"] = out.get("motion_dist", final["motion_dist"])

    return final

def _canonical_overlap_weights(seq_len: int, overlap_size: int, device, dtype=torch.float32):
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
    window_size: int = 128,
    overlap_size: int = 32,
    time_keys=("head_betas", "head_contacts", "head_theta", "pose_quat"),
    z_keys=("head_z", "motion_z"),
    eps: float = 1e-8,
):
    """
    EgoAllo-style overlap blending for long sequences:
      - windows: start = 0, stride, 2*stride, ...
      - end = min(start+W, T)  
      - overlap_weights_slice = canonical[:win_len]
      - pred += out * w
      - weights += w
      - pred /= weights

    Returns:
      dict:
        time_keys -> (B, T, ...)
        z_keys    -> (B, T, latent)  
        head_dist/motion_dist 
    """
    assert head.ndim == 3
    B, T, Dh = head.shape

    device = head.device
    W = window_size
    O = overlap_size
    stride = W - O
    assert stride > 0, "overlap_size must be < window_size"

    canonical_w = _canonical_overlap_weights(T, O, device=device, dtype=torch.float32)

    s0, e0 = 0, min(W, T)
    with torch.inference_mode():
        out0 = model.forward_test(head[:, s0:e0, :])

    final = {}

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
            w_slice = canonical_w[None, :win_len, None] 

            out = model.forward_test(
                head[:, start_t:end_t, :],
            )

            # time keys: (B, win_len, ...) * w 
            for k in time_keys:
                pred = out[k]
                w_b = w_slice
                for _ in range(pred.ndim - 3):
                    w_b = w_b.unsqueeze(-1)  # (1, win_len, 1, 1, ...)
                final[k][:, start_t:end_t, ...] += pred * w_b
                final[f"_{k}_w"][:, start_t:end_t, :] += w_slice

            # z keys: (B, latent) -> (B, win_len, latent)
            for k in z_keys:
                z = out[k]  # (B, latent)
                z_t = z.unsqueeze(1).expand(-1, win_len, -1)  # (B, win_len, latent)
                final[k][:, start_t:end_t, :] += z_t * w_slice
                final[f"_{k}_w"][:, start_t:end_t, :] += w_slice

            final["head_dist"] = out.get("head_dist", final["head_dist"])
            final["motion_dist"] = out.get("motion_dist", final["motion_dist"])

    # 4) (pred /= overlap_weights)
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

def eval():
    HDF5_PATH = DataConfig.HDF5_PATH
    FILE_LIST_PATH = DataConfig.FILE_LIST_PATH
    SUBSEQ_LEN = DataConfig.SUBSEQ_LEN
    
    RESULT_IMG_PATH = EvalConfig.RESULT_IMG_PATH
    RESULT_Z_PATH = EvalConfig.RESULT_Z_PATH
    CHECKPOINT_PATH = EvalConfig.CHECKPOINT_PATH
    BATCH_SIZE = EvalConfig.BATCH_SIZE
    NUM_WORKERS = EvalConfig.NUM_WORKERS

    VISUAL_OUTPUT_DIR = Path("./ours")
    VISUAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(CHECKPOINT_PATH)
 
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
            
            'foot_skate': 0.0,
            'jitter_pred': 0.0,
            'jitter_gt': 0.0,
            'mpjve': 0.0,
    }

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            head, _ = input_data_all(batch)
            start_time = time.time()

            # outputs = forward_test_noncausal_first_then_slide(
            #     model=model,
            #     head=head,           # (B, T, Dh)
            #     window_size=SUBSEQ_LEN,
            # )
            
            # outputs = forward_test_noncausal_overlap_egoallo_style(
            #     model=model,
            #     head=head,           # (B, T, Dh)
            #     window_size=SUBSEQ_LEN,
            #     overlap_size=32,
            # )
          
            outputs = model.forward_test(head)

            runtime = time.time() - start_time

            # # outputs 딕셔너리와 batch 객체에서 필요한 데이터 추출
            # T_world_cpf_data = batch.T_world_cpf[0].cpu().numpy()
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
            # # --- 추가 코드 끝 ---

            print("RUNTIME", runtime)
            with open("store_runtime.txt", "a") as f:
                f.write(f"{runtime}\n")
            
            metrics_batch = calculate_metrics(outputs, batch, egoallo_smpl)

            print(f"\n--- Current Batch Metrics ---")
            print(f"  MPJPE: {metrics_batch['mpjpe']:.4f}")
            print(f"  PAMPJPE: {metrics_batch['pampjpe']:.4f}")
            print(f"  GND: {metrics_batch['GND']:.4f}")
            print(f"  T_head: {metrics_batch['T_head']:.4f}")

            print(f"  FS: {metrics_batch['foot_skate']:.4f}")
            print(f"  Jitter pred: {metrics_batch['jitter_pred']:.4f}")
            print(f"  Jitter gt: {metrics_batch['jitter_gt']:.4f}")
            print(f"  MPJVE: {metrics_batch['mpjve']:.4f}")
            print(f"-----------------------------")

            metrics['mpjpe'] += metrics_batch['mpjpe']
            metrics['pampjpe'] += metrics_batch['pampjpe']
            metrics['GND'] += metrics_batch['GND']
            metrics['T_head'] += metrics_batch['T_head']
            metrics['num_samples'] += 1
            
            metrics['foot_skate'] += metrics_batch['foot_skate']
            metrics['jitter_pred'] += metrics_batch['jitter_pred']
            metrics['jitter_gt'] += metrics_batch['jitter_gt']
            metrics['mpjve'] += metrics_batch['mpjve']
            
            print_metrics(metrics)

        print_metrics(metrics)

if __name__ == '__main__':
    eval()
    