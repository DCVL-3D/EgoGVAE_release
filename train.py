import os
import torch
import torch.utils.data

from pathlib import Path
from tqdm import tqdm
from smplx import SMPLH

# Data
from data.amass import AmassHdf5Dataset
from data.dataclass import collate_dataclass
from data.body_model import BodyModel

# Network
from model.network_7 import Head2Motion

# Criterion
from model.loss import Head2MotionComputeLosses

# Utils
from model.utils import input_data, set_seed, input_data_all, save_loss_to_csv
from model.utils_checkpoint import save_checkpoint
from utils.smplh_egoallo import build_smplh_posed_paper, extract_joints_vertices
from utils.from_egoallo import fncsmpl 

# Metric
from model.metric import egoallo_metric_train, print_metrics

# Config
from config.config_v1_7 import *

def train():

    HDF5_PATH = DataConfig.HDF5_PATH
    FILE_LIST_PATH = DataConfig.FILE_LIST_PATH
    SUBSEQ_LEN = DataConfig.SUBSEQ_LEN
    
    CHECKPOINT_DIR = TrainConfig.CHECKPOINT_DIR
    BATCH_SIZE = TrainConfig.BATCH_SIZE
    NUM_WORKERS = TrainConfig.NUM_WORKERS
    LEARNING_RATE = TrainConfig.LEARNING_RATE
    MAX_EPOCHS = TrainConfig.MAX_EPOCHS
    
    set_seed(SEED)
    print(f"✅ Random Seed fixed: {SEED}")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 데이터셋 및 데이터로더 ---
    train_dataset = AmassHdf5Dataset(
        hdf5_path=HDF5_PATH,
        file_list_path=FILE_LIST_PATH,
        splits=("train",),
        subseq_len=SUBSEQ_LEN,
        cache_files=False,
        slice_strategy=TrainConfig.SLICE_STRATEGY,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        collate_fn=collate_dataclass
    )
    
    smplh = SMPLH(
        model_path=SMPLHConfig.MODEL_PATH,
        gender=SMPLHConfig.GENDER,
        num_betas=SMPLHConfig.NUM_BETAS,
        model_type=SMPLHConfig.MODEL_TYPE,
    ).to(device)
    egoallo_smpl = fncsmpl.SmplhModel.load(SMPLHConfig.EGOALLO_MODEL_PATH).to(device)
    model = Head2Motion(head_dim=TrainConfig.INPUT_HEAD_DIM, motion_dim=TrainConfig.INPUT_MOTION_DIM, out_dim=TrainConfig.OUPUT_MOTION_DIM).to(device)
    
    criterion = Head2MotionComputeLosses(
        lambda_keypoints=TrainConfig.LAMBDA_KEYPOINTS,
        lambda_theta=TrainConfig.LAMBDA_THETA,
        lambda_kl=TrainConfig.LAMBDA_KL,
        lambda_z=TrainConfig.LAMBDA_Z,
        lambda_betas=TrainConfig.LAMBDA_BETAS,
        lambda_contacts=TrainConfig.LAMBDA_CONTACTS,
        lambda_velocity=TrainConfig.LAMBDA_VELOCITY,
        lambda_skating=TrainConfig.LAMBDA_SKATING,
        kl_anneal_end_step=TrainConfig.KL_ANNEAL_END_STEP
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1E-4)
    metrics = {
            'mpjpe': 0.0,
            'pampjpe': 0.0,
            'num_samples': 0,
    }
    # --- 학습 루프 ---
    global_step = 1
    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=True)

        for i, batch in enumerate(progress_bar):

            batch = batch.to(device)
            head, motion = input_data_all(batch)
            
            optimizer.zero_grad()

            outputs = model(head, motion)
            head_posed, motion_posed, gt_posed, head_betas, motion_betas = build_smplh_posed_paper(outputs, batch, egoallo_smpl)
            
            head_joints, _ = extract_joints_vertices(head_posed, head_betas, smplh)
            motion_joints, _ = extract_joints_vertices(motion_posed, motion_betas, smplh)
            gt_joints, _ = extract_joints_vertices(gt_posed, batch.betas, smplh)
            
            loss_dict = criterion(outputs, motion, head_joints, motion_joints, gt_joints, batch, global_step)
            loss = loss_dict['loss_total']
            
            loss.backward()
            optimizer.step()
            global_step += 1

            running_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())
            
            save_loss_to_csv('./train_log_2_6.csv', loss_dict)
            
            
            

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished, Average Loss: {epoch_loss:.4f}")

        checkpoint_path = Path(CHECKPOINT_DIR) / f"epoch_{epoch+1}.pth"
        save_checkpoint(epoch, model, optimizer, loss, checkpoint_path)

if __name__ == '__main__':
    train()
    