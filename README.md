# 🧠 RSNA Intracranial Aneurysm Detection

A deep learning project built for the **RSNA Intracranial Aneurysm Detection** competition on Kaggle.  
This repository explores **3D medical image classification** for detecting intracranial aneurysms in **CT angiography (CTA)** scans, using a pipeline designed for **accuracy, efficiency, and reproducibility**.

---

## 🚀 Project Overview

The goal of this project is to automatically identify **aneurysms** within 3D DICOM series of the brain.  
Each sample represents a full CTA volume, and the model must determine whether an aneurysm is present.

This is a **highly imbalanced** classification problem — only a small subset of patients have positive findings — which makes **data sampling and augmentation** key to achieving strong results.

---

## 🧩 Dataset

- Source: **[RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)**
- Data format: DICOM (3D CT angiography)
- Accompanying CSVs:  
  - `train.csv`: patient-level aneurysm labels  
  - `train_localizers.csv`: slice indices and localization metadata  
  - Segmentation masks for annotated aneurysms  

The dataset is large, and pre-processing is crucial.  
I used **optimized DICOM parsing**, **volume normalization**, and **windowing** (brain window) to extract relevant signal intensity for model input.

---

## ⚙️ Technical Stack

- **Language:** Python 3.10  
- **Framework:** PyTorch (with MONAI transforms)  
- **Environment:** Kaggle Notebook & Local GPU  
- **Tools:** NumPy, pandas, nibabel, tqdm, scikit-learn, Matplotlib  
- **Augmentation:** Random flips, rotations, intensity normalization  
- **Hardware:** A100 / RTX local GPU  

---

## 🧠 Model Design

This project evolved from a baseline **3D CNN** to a **hybrid architecture** optimized for both speed and accuracy.

### Versions Explored:
1. **3D ResNet18 baseline**  
   - Trained on preprocessed 3D patches  
   - BCEWithLogits loss + class weighting

2. **Stratified Oversampling + Focal Loss**  
   - Increased exposure to rare positive cases  
   - Improved sensitivity without overwhelming the model

3. **Patch-based Ensemble (WIP)**  
   - Aggregates predictions from multiple 3D crops per scan  
   - Aims to boost stability and AUC

---

## 📈 Results & Metrics

| Experiment | Model | Sampling | AUC | Notes |
|-------------|--------|-----------|-----|-------|
| v1 | 3D ResNet18 | None | 0.74 | Baseline |
| v2 | 3D ResNet18 | Stratified oversampling | 0.79 | Reduced class imbalance |
| v3 | 3D ResNet34 + Focal Loss | Yes | 0.82 | Better sensitivity on rare positives |

> Metrics are computed using **validation folds** split by patient ID to prevent data leakage.

---

## 🧪 Training Pipeline

`train.py` manages the full training lifecycle:
- DICOM loading → 3D volume assembly  
- Windowing and normalization  
- Augmentations  
- Batch generation with custom `Dataset` and `DataLoader`  
- Multi-GPU training support  
- Fold-based validation and AUC tracking  

```bash
python train.py --epochs 20 --batch-size 8 --fold 0 --model resnet3d34
