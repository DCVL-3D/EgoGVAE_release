<h1 align="center">EgoGVAE: Ego-body Mesh Reconstruction via<br>Guided Variational Autoencoder</h1>


Official Pytorch implementation **"EgoGVAE: Ego-body Mesh Reconstruction via Guided Variational Autoencoder"** <br>
[Jaehun Jung](https://github.com/jaehun00) and [Wonjun Kim](https://sites.google.com/view/dcvl) (Corresponding Author) <br>
🏙️***European Conference on Computer Vision (ECCV)***, Sep. 2026.🏙️

<p align="center"><img src='figures/results.gif'></p>
<p align="center"><img src='figures/overall_architecture.png'></p>
<p align="center">[ Overall architecture ]</p>

## :eyes: Overview
We propose a simple yet powerful method, **EgoGVAE**, for full-body mesh reconstruction from only the head pose of the wearer.

**EgoGVAE** leverages the latent space of the motion-to-motion network, which is a variational autoencoder that takes full-body poses as inputs, to **guide the head-to-motion network**. <br>
This design scheme, which operates with one-step sampling in inference, makes **EgoGVAE perform very fast** compared to diffusion-based approaches.

We provide:

- ✅ **Full implementation** of EgoGVAE

## ✅ Full implementation
### 🐍 Clone the repository
```bash
git clone https://github.com/brentyi/egoallo.git
```

### 📦 Install Dependencies
We provide an installation using Conda package and environment management:
```bash
cd EgoGVAE_release
pip install -e .
```
### 🛠️ Data Preparation & Preprocessing
To run the EgoGVAE inference and training, you need to download the standard human body models.

#### 1. Download the SMPL-H model file
   - Go to the official [MANO website](https://mano.is.tue.mpg.de/).
   - Navigate to the **Download** section and download the file named: 
     `Extended SMPL+H model (used in AMASS project)`
   - Extract the downloaded file and place the `.npz` model files into the `./data/smplh/` directory. 
     *(Example path: `./data/smplh/neural/model.npz`)*
     
2. 🏃‍♂️ **Download the AMASS dataset (Optional for training).**
   - Go to the [AMASS website](https://amass.is.tue.mpg.de/).
   - Download the required motion sequences and place them in `./data/amass/`.

#### 3. Run Initial Preprocessing (Step 1)
Once the downloads are complete, run the first script to process the raw AMASS data with the SMPL-H model:
```bash
python 0a_preprocess_training_data.py --data-root /path/to/amass --smplh-root ./data/smplh

