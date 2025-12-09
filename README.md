# Continuous Global Power Line Localization via Local Polar-Coordinate Feature Learning


## 🚀 Network Architecture

<img src="architecture.jpg">

PLD-Net consists of:
- Grid-wise Polar Representation — Divides the image into grid cells and encodes local line geometry in polar coordinates.
- ConvNeXt-mini Backbone — A lightweight backbone that enlarges the effective receptive field for structural line features.
- Feature Enhancement Modules (WGA & UND) — Strengthen thin-line responses and suppress background noise.
- Prediction Head — Outputs classification scores and polar regression parameters for each grid cell.
- Hungarian Matching — Provides one-to-one supervision between predictions and ground truth during training.
- Global Polar Reconstruction — Converts cell-wise predictions into global line segments during inference.
- Line Integration — Merges and stitches segments to produce continuous power lines.

---

## 📊 Quantitative Comparison with State-of-the-Art

The following table shows our comparison on the **PLDU dataset** and **TTPLA dataset**.

### **Results on PLDU & TTPLA**

| Dataset | Method | UNet | Swin-Unet | DDRNet | MADG-Net | MiT-Unet | SegFormer | **PLD-Net** |
|--------|--------|------|-----------|--------|-----------|-----------|-----------|-------------|
| **PLDU** | F1  | 87.51 | 84.95 | 87.77 | 88.80 | 88.99 | 86.88 | **89.43** |
|         | IoU | 77.64 | 73.12 | 78.12 | 79.55 | 79.73 | 76.12 | **82.04** |
| **TTPLA** | F1 | 77.95 | 77.75 | 77.93 | 78.43 | 82.50 | 72.40 | **82.74** |
|          | IoU | 61.33 | 61.36 | 61.48 | 63.66 | 67.98 | 53.42 | **69.27** |
| **Params (M)** | — | 31.04 | 213.96 | 20.17 | 31.22 | 19.27 | 3.71 | 49.77 |
| **FLOPs (G)**  | — | 218.95 | 180.32 | 17.97 | 39.76 | 73.25 | 6.76 | 27.44 |

---

## 🖼️ Visualization Results

<img src="result-pldu.jpg" height="520">

Note: Visualization of segmented results on PLDU dataset.

<img src="result-ttpla.jpg" height="520">

Note: Visualization of segmented results on TTPLA dataset.

---

## 📁 Project Structure
We only provide partial code for testing purposes.
The available files and structure are as follows:
```text
PLD-Net/
│
├── checkpoints/
│ ├── bestpldu.pth
│ └── bestttpla.pth
│
├── datasets/
│ ├── pldu/
│ └── ttpla/
│
├── models/
│ ├── PLDNet.py
│ ├── UND.py
│ └── WaveletAttention.py
│
├── utils/
│ ├── postprocessPLDU.py 
│ ├── postprocessTTPLA.py.
│ └── logger.py

├── demo.py
├── requirements.txt
└── README.md
```
we place the best-performing training weights in the GitHub Releases section instead of the main code directory.
You can download the pretrained models using the following link:
```bash
https://github.com/User10259/PLD-Net/releases/tag/v1.0-weights
```

## 📦 Requirements

Please install the following packages:
- torch
- torchvision
- timm
- numpy
- opencv-python
- Pillow
- shapely
- albumentations
- matplotlib
- tqdm
- thop
- scipy
- scikit-image

You may create a virtual environment:
```bash
conda create -n pldnet python=3.8
conda activate pldnet
pip install -r requirements.txt
```


## ▶️ Test (Demo Inference)

We provide the best-performing training weights for PLDU and TTPLA.
Use the following command to run inference on a single image:
```bash
python demo.py --dataset pldu --img datasets/pldu/1.jpg
```
or for TTPLA:
```bash
python demo.py --dataset ttpla --img datasets/ttpla/08_1830.jpg
```
The results will be saved under outputs. Modify the image path as needed to test your own images.
