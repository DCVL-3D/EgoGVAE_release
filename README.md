<h1 align="center">EgoGVAE: Ego-body Mesh Reconstruction via<br>Guided Variational Autoencoder</h1>


Official Pytorch implementation **"EgoGVAE: Ego-body Mesh Reconstruction via Guided Variational Autoencoder"** <br>
[Jaehun Jung](https://github.com/jaehun00) and [Wonjun Kim](https://sites.google.com/view/dcvl) (Corresponding Author) <br>
🏙️***European Conference on Computer Vision (ECCV)***, Sep. 2026.🏙️

<p align="center"><img src='figures/results.gif'></p>
<p align="center"><img src='figures/overall_architecture.png'></p>
<p align="center">[ Overall architecture ]</p>

## :eyes: Overview 
We propose a simple yet novel variational method, **EgoGVAE**, for ego-body mesh reconstruction.

By enforcing latent distributions of the motion-to-motion network and the head-to-motion network to be similar, EgoGVAE can easily understand the process of the head-to-motion generation and successfully generate full-body meshes with natural poses.
This design scheme, which operates with one-step sampling in inference, makes EgoGVAE perform very fast compared to diffusion-based approaches.

We provide:

- 🚀 **Minimal plug-and-play code snippet** for quick integration
- ✅ **Full implementation** of DropGaussian
