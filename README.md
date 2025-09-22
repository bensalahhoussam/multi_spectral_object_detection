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
