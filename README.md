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

## YOLO fusion model summary with Ultralytics 

We modified the Ultralytics YOLO framework to support dual input streams (RGB + Thermal).
Each stream passes through its own backbone, and features are fused at multiple levels before detection

| From | n | Module | Arguments |
|------|---|--------|---------|
| -1 | 1 | Conv | [3, 48, 3, 2] |
| -1 | 1 | Conv | [48, 96, 3, 2] |
| -1 | 2 | C2f | [96, 96, 2, True] |
| -1 | 1 | Conv | [96, 192, 3, 2] |
| -1 | 4 | C2f | [192, 192, 4, True] |
| -6 | 1 | Conv | [3, 48, 3, 2] |
| -1 | 1 | Conv | [48, 96, 3, 2] |
| -1 | 2 | C2f | [96, 96, 2, True] |
| -1 | 1 | Conv | [96, 192, 3, 2] |
| -1 | 4 | C2f | [192, 192, 4, True] |
| [4, 9] | 1 | GPT | [192] |
| [4, 10] | 1 | Add2 | [192, 0] |
| [9, 10] | 1 | Add2 | [192, 1] |
| 11 | 1 | Conv | [192, 384, 3, 2] |
| -1 | 4 | C2f | [384, 384, 4, True] |
| 12 | 1 | Conv | [192, 384, 3, 2] |
| -1 | 4 | C2f | [384, 384, 4, True] |
| [14, 16] | 1 | GPT | [384] |
| [14, 17] | 1 | Add2 | [384, 0] |
| [16, 17] | 1 | Add2 | [384, 1] |
| 18 | 1 | Conv | [384, 768, 3, 2] |
| -1 | 2 | C2f | [768, 768, 2, True] |
| -1 | 1 | SPPF | [768, 768, 5] |
| 19 | 1 | Conv | [384, 768, 3, 2] |
| -1 | 2 | C2f | [768, 768, 2, True] |
| -1 | 1 | SPPF | [768, 768, 5] |
| [22, 25] | 1 | GPT | [768] |
| [22, 26] | 1 | Add2 | [768, 0] |
| [25, 26] | 1 | Add2 | [768, 1] |
| [11, 12] | 1 | Add | [192] |
| [18, 19] | 1 | Add | [384] |
| [27, 28] | 1 | Add | [768] |
| -1 | 1 | Upsample | [None, 2, 'nearest'] |
| [-1, 30] | 1 | Concat | [1] |
| -1 | 2 | C2f | [1152, 384, 2] |
| -1 | 1 | Upsample | [None, 2, 'nearest'] |
| [-1, 29] | 1 | Concat | [1] |
| -1 | 2 | C2f | [576, 192, 2] |
| -1 | 1 | Conv | [192, 192, 3, 2] |
| [-1, 34] | 1 | Concat | [1] |
| -1 | 2 | C2f | [576, 384, 2] |
| -1 | 1 | Conv | [384, 384, 3, 2] |
| [-1, 31] | 1 | Concat | [1] |
| -1 | 2 | C2f | [1152, 768, 2] |
| [37, 40, 43] | 1 | Detect | [4, [192, 384, 768]] |


### Feature Dimensions

The model processes features at three different scales:
- **192 channels** (shallow features)
- **384 channels** (mid-level features)
- **768 channels** (deep features)

### Fusion Strategy

The architecture employs a dual-branch processing approach with:
1. Parallel processing of inputs through two similar backbone structures, first layer and layer 6 
2. Multiple fusion points using GPT and Add2 modules
3. Hierarchical feature aggregation at different network depths

This design enables robust feature extraction and integration, making it suitable for complex computer vision tasks.


## Demo 

![Uploading ezgif-41d2a3f8e1588a.gif…]()


## Data Preparation 

This project requires paired RGB and Thermal images along with their corresponding labels. The dataset is organized into three splits: train, val, and test.

### 📂 File structure

- dataset/
  - images/
    - train/         
    - val/           
    - test/           
  - thermal/
    - train/        
    - val/         
    - test/         
  - labels/
    - train/        
    - val/            
    - test/


  ## Model Training
```bash
!yolo detect Dualtrain model=yolov_fusion.yaml data=FLIR.yaml workers=2 batch=12 device=0 epochs=30 patience=80 name=visible
```

