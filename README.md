# Multispectral Object Detection for Autonomous Driving in Night Scenes

This project focuses on improving object detection in challenging night-time environments by leveraging multispectral data. Standard RGB cameras often struggle in low-light conditions, leading to poor detection accuracy and reduced safety in autonomous driving. To overcome this limitation, we integrate RGB and thermal (infrared) imagery for robust perception at night.

## Key Contributions

Dual-Input Model: Modified the Ultralytics YOLO framework to process two input modalities (RGB + Thermal) simultaneously.

Enhanced Detection: Fuses complementary information from visible and thermal spectra, enabling reliable detection of pedestrians, vehicles, and other objects in night scenes.

Adaptation of Ultralytics Framework: Customized data loading, model architecture, and training pipeline to support multispectral input without breaking compatibility with YOLO’s ecosystem.

## Applications

Autonomous driving in night-time and low-visibility scenarios

Surveillance and safety systems

General-purpose multispectral vision research


## Model Overview

<img width="5120" height="1974" alt="image" src="https://github.com/user-attachments/assets/b6ef4233-1279-4a1b-856b-f5325b5124cc" />

## Data Preparation 

This project requires paired RGB and Thermal images along with their corresponding labels. The dataset is organized into three splits: train, val, and test.

FusionGAN
├── ...
├── Test_LLVIP_ir
|   ├── 190001.jpg
|   ├── 190002.jpg
|   └── ...
├── Test_LLVIP_vi
|   ├── 190001.jpg
|   ├── 190002.jpg
|   └── ...
├── Train_LLVIP_ir
|   ├── 010001.jpg
|   ├── 010002.jpg
|   └── ...
└── Train_LLVIP_vi
    ├── 010001.jpg
    ├── 010002.jpg
    └── ...
