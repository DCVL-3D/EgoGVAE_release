<h1 align="center">EgoGVAE: Ego-body Mesh Reconstruction via<br>Guided Variational Autoencoder</h1>


Official Pytorch implementation **"EgoGVAE: Ego-body Mesh Reconstruction via Guided Variational Autoencoder"** <br>
[Jaehun Jung](https://github.com/jaehun00) and [Wonjun Kim](https://sites.google.com/view/dcvl) (Corresponding Author) <br>
🏙️***European Conference on Computer Vision (ECCV)***, Sep. 2026.🏙️

<p align="center"><img src='figures/results.gif'></p>
<p align="center"><img src='figures/overall_architecture.png'></p>
<p align="center">[ Overall architecture ]</p>

## :eyes: Overview
We propose a simple yet powerful method, **EgoGVAE**, for full-body mesh reconstruction from only the head pose of the wearer.

**EgoGVAE** leverages the latent space of the motion-to-motion network, which is a variational autoencoder that takes full-body poses as inputs, **to guide the head-to-motion network**. <br>
This design scheme, which operates with one-step sampling in inference, makes **EgoGVAE perform very fast** compared to diffusion-based approaches.

## 📦 Environment Setup & Install Dependencies
We provide an installation using Conda package and environment management:
```bash
git clone https://github.com/DCVL-3D/EgoGVAE_release.git
cd EgoGVAE_release
pip install -e .
```
## 🛠️ Data Preparation & Preprocessing
To run the EgoGVAE inference and training, please prepare the SMPL model and preprocess the dataset as follows:

1. **Download the SMPL-H model file**
   - Go to the official [MANO website](https://mano.is.tue.mpg.de/).
   - Navigate to the **Download** section and you can download the file named: <br>
     `Extended SMPL+H model (used in AMASS project)`
   - Extract the downloaded file and place the `.npz` model files into the `./data/smplh/` directory. <br>
     *(Example path: `./data/smplh/neural/model.npz`)*
     
2. **Download the AMASS dataset (Optional for training).**
   - Download the required motion sequences from the [AMASS website](https://amass.is.tue.mpg.de/).
   - Place the downloaded files in the `./data/amass/` directory.

3. **Preprocess the dataset.**
   - To preprocess the downloaded data, we provide two python scripts.
     ```bash
     python ./data/preprocess/preprocess_step1.py --data-root /path/to/amass --smplh-root ./data/smplh
     ```
     ```bash
     python ./data/preprocess/preprocess_step2.py --data-npz-dir ./data/processed_30fps_no_skating/
     ```
   > 💡 **Note:** For more detailed information regarding the data preprocessing pipeline, please refer to the [EgoAllo GitHub repository](https://github.com/brentyi/egoallo).

## ⚡ Run Inference
You can download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1SBX7KSM8PKOcrYbnUAhXB-rCPgMR47Ac?usp=drive_link). <br>
After downloading, please place model weights (`.pth` files) in the `./outputs/pre-trained` folder.

#### ⚙️ Path Configuration
Before running the evaluation, please update the dataset and model paths in `./config/config.py` to match your local environment:
- `HDF5_PATH` & `FILE_LIST_PATH`: Paths to your preprocessed dataset files.
- `MODEL_PATH`: Path to SMPL model files.
- `CHECKPOINT_DIR`: Directory containing the pre-trained model weights (e.g., `./outputs/pre-trained`).

#### 🚀 Running Evaluation
Once paths are configured, you can run the evaluation script:

   ```bash
   # Activate the environment
   conda activate egogvae

   # Run the evaluation script
   python eval.py
   ```

#### 🎥 Visualization
1. **Merging Outputs**
  - After the evaluation is complete, you can merge output `.npz` files into a single format using the provided script: 

   ```bash
   python utils/convert_npz_to_p.py \
       --input-dir ./outputs/results \
       --output-dir ./outputs/merged_results
   ```

2. **Visualize Outputs**
   - We provide a script to visualize the merged results in 3D. 
   - Our visualization tool is built upon [viser](https://github.com/nerfstudio-project/viser).

   - To launch the visualization server, please run:

   ```bash
   python utils/visualization.py \
       --data-root-dir ./outputs/merged_results \
       --smplh-npz-path ./data/smplh/neural/model.npz
   ```

## 🏋️ Training
```bash
# Activate the environment
conda activate egogvae

# Run the training script
python train.py
```

## Acknowledgments
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25471545).

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[brentyi/egoallo](https://github.com/brentyi/egoallo)  </br>
[Mathux/TEMOS](https://github.com/Mathux/TEMOS)  </br>

## Citation
If you find our work useful for your project, please consider citing the following paper.
