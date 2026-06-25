from pathlib import Path

PROJECT_NAME = "egotext"
SEED = 2025

class DataConfig:
    HDF5_PATH = Path("/home/dev2/Drive_C/KHW/_egotext/head2motion/dataset/rich_final/RICH_FINAL_0928.hdf5")
    FILE_LIST_PATH = Path("/home/dev2/Drive_C/KHW/_egotext/head2motion/dataset/rich_final/RICH_FINAL_0928_files.txt")
    # HDF5_PATH = Path("/home/dev2/Drive_C/KHW/_egotext/head2motion/dataset/rich_final/RICH_test_ransac.hdf5")
    # FILE_LIST_PATH = Path("/home/dev2/Drive_C/KHW/_egotext/head2motion/dataset/rich_final/RICH_test_ransac_files.txt")
    SUBSEQ_LEN = 128
    
class SMPLHConfig:
    MODEL_PATH = "/home/dev2/Drive_C/KHW/TokenHMR/_ego/lib/data/body_models/smplh"
    GENDER = "NEUTRAL"
    NUM_BETAS = 16
    MODEL_TYPE = "smplh"
    EGOALLO_MODEL_PATH = "/home/dev2/Drive_C/KHW/egoallo/data/smplh/neutral/model.npz"
    
class TrainConfig:
    LEARNING_RATE = 1E-4
    MAX_EPOCHS = 1000000
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    SLICE_STRATEGY = "random_uniform_len"
    
    CHECKPOINT_DIR = './outputs/v_re'
    
    # Input, Output dimension
    INPUT_HEAD_DIM = 16
    INPUT_MOTION_DIM = 135
    OUPUT_MOTION_DIM = 126 + 16 + 21
    
    # Loss Weight
    LAMBDA_KEYPOINTS = 1.0
    LAMBDA_THETA = 1.0
    LAMBDA_KL = 1.0E-2
    LAMBDA_Z = 1.0E-2
    # LAMBDA_Z = 1.0E-3
    # LAMBDA_Z = 0.0
    LAMBDA_BETAS = 1.0E-3
    LAMBDA_CONTACTS = 1.0E-3
    LAMBDA_VELOCITY = 1.0E-3
    LAMBDA_SKATING = 0.0
    
    KL_ANNEAL_END_STEP = 1
    
    # Flag
    FLAG_JOINT_LOSS = True
    FLAG_VERTICE = False

class EvalConfig:
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    
    SLICE_STRATEGY = "deterministic"
    
    CHECKPOINT_PATH = "./outputs/v_re/_latest.pth"
    
    
    RESULT_IMG_PATH = './image_eval.png'
    RESULT_Z_PATH = './image_z.png'
    
    # Input, Output dimension
    INPUT_HEAD_DIM = 16
    INPUT_MOTION_DIM = 135
    OUPUT_MOTION_DIM = 126 + 16 + 21

class VisConfig:
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    
    SLICE_STRATEGY = "deterministic"
    
    CHECKPOINT_PATH = "./outputs/checkpoints_paper_jjh_v3/_latest.pth"
    ANIMATION_DIR = "./outputs/animations_paper_jjh_v3_amass"
    
    # Input, Output dimension
    INPUT_HEAD_DIM = 16
    INPUT_MOTION_DIM = 135
    OUPUT_MOTION_DIM = 126 + 16 + 21
    
    # Flag
    FLAG_VERTICE = True