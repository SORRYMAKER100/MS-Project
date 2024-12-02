# Wide-Field, High-Resolution Reconstruction in Computational Multi-Aperture Miniscope using CoordGate Network

## Overview
This project leverages a Microlens Array (MLA)-based optical system to reconstruct high-quality images over a large field of view (FOV), addressing challenges such as image overlap and blurriness. By employing a neural network-based approach incorporating the CoordGate module, this study enhances the spatially variant (SV) deblurring process, achieving high-resolution 2D reconstructions of ground truth objects. The model balances computational efficiency and accuracy, with future applications in 3D deconvolution tasks.

## Problem Statement
Microscopy systems often suffer from spatially variant distortions and blurring due to optical limitations, particularly at the edges of the FOV. These challenges are exacerbated in MLA-based systems where unique distortions arise from each lenslet. Standard convolutional neural networks (CNNs) fail to handle these localized distortions due to their spatially invariant nature.

To address these challenges:
- **CoordGate** is introduced to incorporate pixel and lenslet coordinates into the convolutional process, enabling linear shift-variant (LSV) convolutions.
- **Positional Encoding** enhances the model's ability to capture complex spatial relationships without significant computational overhead.

## Key Components

### Model Architecture
- **Base Model:** Modified UNet or ResNet architecture with encoder-decoder structure for multi-scale feature extraction.
- **CoordGate Module:** Incorporates positional information (pixel and lens coordinates) into convolution operations to enable LSV processing.
- **Transformer Integration:** Lightweight Transformer components enhance global spatial dependency understanding.
- **Positional Encoding Techniques:**
  - Sinusoidal Encoding for periodic spatial information.
  - Radial Basis Functions (RBF) for flexible non-linear mappings.

### Data Processing
- **Input:** Nine images from MLA corresponding to distinct lenslets, each with specific distortions.
- **Patch-Wise Processing:** Improves memory efficiency by focusing on localized image details.

### Loss and Metrics
- **Loss Function:** Binary Cross-Entropy (BCE) for pixel-wise deblurring tasks.
- **Evaluation Metrics:**
  - Peak Signal-to-Noise Ratio (PSNR)
  - Structural Similarity Index (SSIM)

### Framework and Tools
- **Programming Language:** Python 3.8
- **Deep Learning Library:** PyTorch 2.4
- **Hardware:** GPU-accelerated training for high-resolution inputs.

## Result
![image](https://github.com/user-attachments/assets/22cecb7c-4dc0-4008-bb52-001c1d6997a8)

(a) is the Non-linear Free Network provided by the paper Simple Baselines for Image Restoration, consisting of a model with 36 hidden layers. The reconstructed image shows blurred or even missing edge information, which demonstrates that a neural network model based on traditional CNNs cannot accurately predict different point spread functions (PSFs) for different patches without additional positional information.

(b) shows the results generated by a network model enhanced with a CoordConv.  The final results are significantly better than those in (a). However, the generated image still has blurred edge regions, with particular shortcomings in restoring high-frequency details.

(c) shows the results of using the CoordGate. The mask generated by the Coordinate MLP flexibly adjusts the CNN's output, enabling the neural network to learn shift-variant PSFs. At the same time, I reduced the number of hidden layers in the model and used grouped convolutions to decrease the number of parameters.

(d) is the ground truth.

![image](https://github.com/user-attachments/assets/07927a10-4c29-4aa5-adac-2e33047b3b41)


## Literature Review
- **Deconvolution Methods:** Semi-blind methods using training datasets to estimate local PSF variations for MLA systems.
- **CoordGate vs CoordConv:** CoordGate enhances convolutional flexibility for spatially variant data compared to CoordConv.
- **Transformers in Vision:** Potential for long-range spatial modeling but requires lightweight adaptations for efficient deblurring.

## References
1. M. Tancik et al., "Fourier features let networks learn high frequency functions in low dimensional domains," 2020.
2. M. Hirsch et al., "Efficient filter flow for space-variant multiframe blind deconvolution," 2010.
3. F. Soulez, "A 'learn 2D, apply 3D' method for 3D deconvolution microscopy," 2014.
4. M. Buzzoni et al., "Blind deconvolution based on cyclostationarity maximization," 2018.
5. A. Shajkofci et al., "Semi-blind spatially-variant deconvolution in optical microscopy," 2018.
6. Y. Xue et al., "Computational Miniature Mesoscope V2," 2022.
7. R. Liu et al., "An intriguing failing of convolutional neural networks and the coordconv solution," 2018.
8. S. Howard et al., "CoordGate: Efficiently Computing Spatially-Varying Convolutions," 2024.
9. D. Alexey et al., "An image is worth 16x16 words: Transformers for image recognition at scale," 2020.
