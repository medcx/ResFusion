<div align="center">
  
<h1> Medical Image Fusion Driven by Implicit-Forward Diffusion and Time-aware Joint Optimization </h1>

</div>
<div>
    <h4 align="center">
        <a href="https://github.com/medcx/ResFusion" target='_blank'>
        <img src="https://img.shields.io/badge/Github%20🤗-ResFusion-yellow">
        </a>
        <img src="https://visitor-badge.laobi.icu/badge?page_id=medcx/ResFusion">
        <a href="https://github.com/medcx/ResFusion" target='_blank'>
        <img src="https://img.shields.io/github/stars/medcx/ResFusion?style=social">
        </a>
    </h4>
</div>

## 🔑Caption
This is the codebase for article ResFusion: Medical Image Fusion Driven by Implicit-Forward Diffusion and Time-aware Joint Optimization

## 🔧Quick Start
**Demo**

We provide the pre-trained model for BraTs dataset, please save it to ```demo_results/brats_demo.pth```. 

Here are the download link: 
[ResFusion_BraTs.pth](https://drive.google.com/file/d/1i_Qt28X4c7QUhAvQKeSkB8eOJGSOIiGa/view?usp=sharing)

Run the following code for the demo result:
```
python test_demo.py
```


**Training**

Run the following code to train the model on BraTs dataset:
```
python main.py --cfg_path configs/fusion_brats.yaml --save_dir results/brats
```
