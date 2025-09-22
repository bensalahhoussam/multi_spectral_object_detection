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

                   from  n    params  module                                       arguments                     
  0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]                 
  1                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]                
  2                  -1  2    111360  ultralytics.nn.modules.block.C2f             [96, 96, 2, True]             
  3                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  4                  -1  4    813312  ultralytics.nn.modules.block.C2f             [192, 192, 4, True]           
  5                  -6  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]                 
  6                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]                
  7                  -1  2    111360  ultralytics.nn.modules.block.C2f             [96, 96, 2, True]             
  8                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  9                  -1  4    813312  ultralytics.nn.modules.block.C2f             [192, 192, 4, True]           
 10              [4, 9]  1   3583872  ultralytics.nn.modules.block.GPT             [192]                         
 11             [4, 10]  1         0  ultralytics.nn.modules.block.Add2            [192, 0]                      
 12             [9, 10]  1         0  ultralytics.nn.modules.block.Add2            [192, 1]                      
 13                  11  1    664320  ultralytics.nn.modules.conv.Conv             [192, 384, 3, 2]              
 14                  -1  4   3248640  ultralytics.nn.modules.block.C2f             [384, 384, 4, True]           
 15                  12  1    664320  ultralytics.nn.modules.conv.Conv             [192, 384, 3, 2]              
 16                  -1  4   3248640  ultralytics.nn.modules.block.C2f             [384, 384, 4, True]           
 17            [14, 16]  1  14245632  ultralytics.nn.modules.block.GPT             [384]                         
 18            [14, 17]  1         0  ultralytics.nn.modules.block.Add2            [384, 0]                      
 19            [16, 17]  1         0  ultralytics.nn.modules.block.Add2            [384, 1]                      
 20                  18  1   2655744  ultralytics.nn.modules.conv.Conv             [384, 768, 3, 2]              
 21                  -1  2   7084032  ultralytics.nn.modules.block.C2f             [768, 768, 2, True]           
 22                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 23                  19  1   2655744  ultralytics.nn.modules.conv.Conv             [384, 768, 3, 2]              
 24                  -1  2   7084032  ultralytics.nn.modules.block.C2f             [768, 768, 2, True]           
 25                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 26            [22, 25]  1  56802816  ultralytics.nn.modules.block.GPT             [768]                         
 27            [22, 26]  1         0  ultralytics.nn.modules.block.Add2            [768, 0]                      
 28            [25, 26]  1         0  ultralytics.nn.modules.block.Add2            [768, 1]                      
 29            [11, 12]  1         0  ultralytics.nn.modules.block.Add             [192]                         
 30            [18, 19]  1         0  ultralytics.nn.modules.block.Add             [384]                         
 31            [27, 28]  1         0  ultralytics.nn.modules.block.Add             [768]                         
 32                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 33            [-1, 30]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 34                  -1  2   2067456  ultralytics.nn.modules.block.C2f             [1152, 384, 2]                
 35                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 36            [-1, 29]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 37                  -1  2    517632  ultralytics.nn.modules.block.C2f             [576, 192, 2]                 
 38                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]              
 39            [-1, 34]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 40                  -1  2   1846272  ultralytics.nn.modules.block.C2f             [576, 384, 2]                 
 41                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 42            [-1, 31]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 43                  -1  2   7378944  ultralytics.nn.modules.block.C2f             [1152, 768, 2]                
 44        [37, 40, 43]  1   4220380  ultralytics.nn.modules.head.Detect           [4, [192, 384, 768]]          
YOLOv_fusion summary: 554 layers, 124,850,236 parameters, 124,850,220 gradients, 3873.8 GFLOPs












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
