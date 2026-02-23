# Fetal Brain MRI Segmentation Benchmark Framework

This repository provides a PyTorch-based benchmarking framework for fetal brain tissue segmentation from MRI scans.

The framework includes:

- Proposed model (ResNet-34 + Lightweight Decoder)
- Multiple baseline segmentation models
- Cross-validation training
- Standardized evaluation metrics
- Modular and extensible design

This repository is designed for research reproducibility and model comparison.

---

## 🔬 Implemented Models

### ✅ Proposed Model
- ResNet-34 encoder
- Custom lightweight decoder

### 📊 Baseline Models Included

The following segmentation models are supported:

- UNet
- UNet++
- DeepLabV3
- DeepLabV3+
- FPN
- PSPNet
- SegFormer
-----
- Custom ONN-based decoders (if enabled)

All models can be selected via configuration.

---

## 📁 Project Structure
Fetal-Brain-Segmentation/
│
├── configs/
│ ├── config.py
│ └── config_test.py
│
├── preprocessing/
│ └── CreateFolds.m
│
├── src/
│ ├── data/
│ │ └── dataset.py
│ │
│ ├── models/
│ │ └── models.py
│ │
│ └── utils/
│ ├── utils.py
│ └── image_mean_std.py
│
├── training/
│ ├── train.py
│ └── test.py
│
├── requirements.txt
└── README.md

---

## 📂 Dataset Format

Expected dataset structure:
Data/
├── Train/
│ ├── fold_1/
│ │ ├── images/
│ │ └── masks/
│
├── Val/
│ ├── fold_1/
│ │ ├── images/
│ │ └── masks/
│
└── Test/
├── fold_1/
├── images/
└── masks/

---

## ⚙️ Configuration

All experiments are controlled via:

You can modify:

- Model type
- Encoder backbone
- Loss function
- Learning rate
- Batch size
- Number of folds
- Decoder type
- Attention mechanism

---

## 🚀 Training

Run from the project root:

You can modify:

- Model type
- Encoder backbone
- Loss function
- Learning rate
- Batch size
- Number of folds
- Decoder type
- Attention mechanism

---

## 🚀 Training

Run from the project root:
training/train.py


---

## 🧪 Evaluation
training/test.py


---

## 📊 Evaluation Metrics

The framework computes:

- Accuracy
- Intersection over Union (IoU)
- Dice Score (DSC)
- Per-class metrics (for multi-class segmentation)

---

## 🔄 Cross-Validation

- Patient-wise fold separation
- Multi-fold training supported
- Average performance reporting

---

## 🖥️ Hardware Support

- CUDA GPU recommended
- Multi-GPU supported
- CPU fallback available

---

## 🧠 Extensibility

Researchers can easily:

- Add new encoders
- Add new decoders
- Implement new loss functions
- Plug in transformer-based models
- Add 3D models

The architecture is modular to support future research extensions.

---

## 📜 License

MIT License

---

## 👨‍🔬 Intended Use

This repository is intended for:

- Academic research
- Model benchmarking
- Reproducible experiments
- Fetal MRI segmentation studies

