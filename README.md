# 🧠 A3-TTA: Adaptive Anchor Alignment Test-Time Adaptation for Image Segmentation

This repository contains the official PyTorch implementation of:

**A3-TTA: Adaptive Anchor Alignment Test-Time Adaptation for Image Segmentation**  
📘 *IEEE Transactions on Image Processing (TIP), 2025*  
🔗 https://ieeexplore.ieee.org/document/11311445

---

## 🔍 Overview

**A3-TTA** is a **source-free test-time adaptation (TTA)** framework for robust image segmentation under domain shift.  
It adapts a pretrained segmentation model **online at test time**, using *only unlabeled target-domain data*, without access to source images or retraining.

Unlike prior pseudo-label-based TTA methods that rely on perturbation ensembles (e.g., dropout, TTA, noise), **A3-TTA introduces anchor-guided alignment**, constructing **stable and distribution-aware pseudo supervision** for reliable adaptation.

A3-TTA is:
- 🔄 **Online & continual** (single-pass over test data)
- ❌ **Source-free**
- 🧩 **Model-agnostic**
- 🧠 **Structure-aware** (boundary-sensitive dense supervision)

---

## 📦 Installation

We recommend using a conda environment:

```bash
conda create -n a3tta python=3.9
conda activate a3tta
pip install -r requirements.txt
```

---

## 📂 Dataset: M&MS

We adopt the **M&MS dataset** for cardiac MRI segmentation.  
Official website: https://www.ub.edu/mnms/

### Steps:

1. Apply for and obtain **official permission** to use the dataset.  
2. Download the dataset through the official portal.  
3. (Optional) Use our **processed M&MS 2D version**:

   👉 Google Drive:  
   https://drive.google.com/file/d/1jaT2nsbF1-rPoWs6fnF9DsFxTuaYXqW2/view?usp=sharing

4. Extract the processed data into: A3-TTA/data/mms2d/

## 📂 Dataset: Prostate
### Steps:

1. Apply for and obtain **official permission** to use the dataset.  
2. Download the dataset through the official portal.  
3. (Optional) Use our **processed Prostate data**:

   👉 Google Drive:  
   https://drive.google.com/file/d/1MdDJqxqiZ_0vYdPmcZMvVr-jcz8dxaYk/view?usp=drive_link 

4. Extract the processed data into: A3-TTA/data/prostate2d/


---

## 🚀 Source Model Training (Using the M&Ms Dataset; Prostate Dataset Follows the Same Procedure)

To train a UNet segmentation model on the source domain:

```bash
python train_source_2d.py --cfg cfgs/prostate/source.yaml
```

- Checkpoints are saved to: `save_model/`

---

## 🧪 Test-Time Adaptation

We provide evaluation scripts for A3-TTA and other methods. All experiments are configured via YAML files.

```bash
# A3-TTA (our method)
python test-time-adaptation.py --cfg cfgs/prostate/a3-tta_prostate.yaml
```


---

## 📚 Citation

If you find A3-TTA useful, please consider citing:

```bibtex
@article{wu2025a3tta,
  author={Wu, Jianghao and Luo, Xiangde and Zhou, Yubo and Wu, Lianming and Wang, Guotai and Zhang, Shaoting},
  journal={IEEE Transactions on Image Processing}, 
  title={A3-TTA: Adaptive Anchor Alignment Test-Time Adaptation for Image Segmentation}, 
  year={2025},
  volume={34},
  pages={8511--8522},
  doi={10.1109/TIP.2025.3644789}
}
```



## 🙋 Acknowledgements

This repo builds upon ideas from:

- [TENT](https://github.com/DequanWang/tent)
- and others, re-implemented for medical segmentation tasks.

---

## 📮 Contact

If you have questions, feel free to open an issue or reach out.

Happy Adapting! 🎯

