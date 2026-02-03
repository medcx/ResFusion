<div align="center">
  
<h1> ResFusion: Medical Image Fusion Driven by Implicit-Forward Diffusion and Time-aware Joint Optimization </h1>

</div>

## 🔑Caption
This is the codebase for article ResFusion: Medical Image Fusion Driven by Implicit-Forward Diffusion and Time-aware Joint Optimization

## 🔧Quick Start
**Demo**

We provide the pre-trained model for BraTs dataset, please save it to ```demo_results/brats_demo.pth```. 

Here are the download link: 
(Coming soon)

Run the following code for the demo result:
```
python test_demo.py
```


**Training**

Run the following code to train the model on BraTs dataset:
```
python main.py --cfg_path configs/fusion_brats.yaml --save_dir results/brats
```
