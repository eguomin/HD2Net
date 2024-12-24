## HD2Net

HD2Net is the companion code to our paper:

**HD2Net：A Deep Learning Framework for Simultaneous Denoising and Deaberration in Fluorescence Microscopy**

This work is based on [DeAbe](https://github.com/eguomin/DeAbePlus)[1] and [3D-RCAN](https://github.com/AiviaCommunity/3D-RCAN)[2].

The code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR)[3].

## Test environment

The training and model application were performed within Python 3.9.18 on a Linux workstation with the following specifications: 

- CPU - Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz
- RAM - 128 GB 
- GPU - NVIDIA GeForce TITAN RTX with 24 GB memory.

## Installation

1. Clone the repo
```
    git clone https://github.com/eguomin/HD2Net.git
```

2. Install dependent packages
```
    cd HD2Net
    pip install -r requirements.txt
```

3. Install

    Please run the following commands in the root path to install:

```
    python setup.py develop
```

## Train

1. Data preparation

    - The overall workflow is similar as DeAbe. We used `scripts\matlab_scripts\test_gen_simu_3Dimage_fromSynObj_HD2Net.m` and `scripts\data_preparation\tif2reds.py` to generate dataset.

    - MATLAB scripts can refer to the usage of DeAbe.

    - The Python script aims to convert TIF images into the REDS dataset format, which can greatly reduce memory usage for datasets that occupy a large amount of storage space.
  
    - The generated `meta_info` file needs to be placed in `basicsr\data\meta_info`
2. Model training

    - The training process is similar to BasicSR. Training parameters can be set in the YAML file.
```
python basicsr/train.py -opt options\train\train_HD2Net_SO.yml 
```

## Test

   - Once the HD2Net model is trained, we can use the following command to apply it to new data.
 ```
 python basicsr/test.py -opt options\test\test_HD2Net_SO.yml 
 ```


## Reference

[1]	M. Guo, Y. Wu, C. M. Hobson, Y. Su, S. Qian, E. Krueger, R. Christensen, G. Kroeschell, J. Bui, M. Chaw, and others, "Deep learning-based aberration compensation improves contrast and resolution in fluorescence microscopy," bioRxiv 2023–10 (2023).

[2] J. Chen, H. Sasaki, H. Lai, Y. Su, J. Liu, Y. Wu, A. Zhovmer, C. A. Combs, I. Rey-Suarez, H.-Y. Chang, and others, "Three-dimensional residual channel attention networks denoise and sharpen fluorescence microscopy image volumes," Nature methods 18, 678–687 (2021).

[3] Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy and Chao Dong. BasicSR: Open Source Image and Video Restoration Toolbox. https://github.com/xinntao/BasicSR, 2022.