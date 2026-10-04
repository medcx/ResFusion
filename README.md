
<div align="center">

<h1>Medical Image Fusion Driven by Implicit-Forward Diffusion and Time-aware Joint Optimization</h1>

<h3>NeurIPS 2026</h3>

<p>
  <a href="https://github.com/medcx/ResFusion"><img src="https://img.shields.io/badge/GitHub-ResFusion-181717?logo=github&logoColor=white" alt="GitHub"></a>&nbsp;
  <img src="https://visitor-badge.laobi.icu/badge?page_id=medcx/ResFusion" alt="Visitors">&nbsp;
  <a href="https://github.com/medcx/ResFusion"><img src="https://img.shields.io/github/stars/medcx/ResFusion?style=social" alt="Stars"></a>
</p>

</div>

---

## 📖 Overview

This repository contains the official implementation of **ResFusion: Medical Image Fusion Driven by Implicit-Forward Diffusion and Time-aware Joint Optimization**, presented at **NeurIPS 2026**.

## 🚀 Quick Start

### 📦 Pre-trained Model

We provide a pre-trained ResFusion model for the BraTS dataset.

Download the checkpoint and place it in the following directory:

`demo_results/brats_demo.pth`

**Download:** [ResFusion_BraTS.pth](https://drive.google.com/file/d/1i_Qt28X4c7QUhAvQKeSkB8eOJGSOIiGa/view?usp=sharing)

### 🔍 Demo

Run the following command to perform inference using the pre-trained model:

```bash
python test_demo.py
```

### 🔥 Training

To train ResFusion on the BraTS dataset, run:

```bash
python main.py \
    --cfg_path configs/fusion_brats.yaml \
    --save_dir results/brats
```

---

## 📜 Citation

If you find our work useful for your research, please consider citing our paper and starring this repository.

---

## ⭐ Acknowledgments

Thank you for your interest in ResFusion!
