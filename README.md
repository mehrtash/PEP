# PEP: Parameter Ensembling by Perturbation

This repository is the official implementation of **PEP: Parameter Ensembling by Perturbation.**

<!-- > 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials
-->

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Project setup
1. Change the project root in the `helpers/settings.py` file to the directory that 
you want to run experiments.
2. Run `settings.py` file to create the folder structure for this project:
    ```
    cd helpers
    python settings.py
    ```
    which should create the following folder structure:
    ```
    ├── intermediate
    │   ├── data
    │   │   ├── arrays
    │   │   └── sheets
    │   └── models
    └── raw
    ```
3. Download the preprocessed ImageNet data (center cropped) that we used in this project from the 
following link:
[Download Link](https://www.dropbox.com/sh/5nwkk693coegsr4/AACRBvEV_1micL5bBnmbPydea?dl=0)

4. copy the folders imagenet_224 and imagenet_299 to the `[project_root]/intermediate/data/arrays/` folder.
These are numpy arrays containing 50,000 validation images from ILSVRC2012.

#### optional preprocessing
Optionally, instead of running steps 3 and 4 you can create these arrays
by runnin the code in `preprocessing` folder. 
First, you have to download the ILSVRC2012_devkit_t12
form ImageNet website, put the JPEG images in `[project_root]/raw/imagenet_validation/`
and then run the following codes for pre-processing:
```
cd preprocessing
python 1_resize_crop_images.py
python 2_preprocess_y_val.py
```

## ImageNet Experiments
### Figure 1

![](assets/figure1.png)

### Finding Optimal Sigmas

### PEP

### Analysis and comparison with Temperature Scaling

## MNIST Experiments

## CIFAR-10 Experiments

## Overfitting Experiments

