 Lane Detection Using a U-Net CNN
 
1. Dataset:
The dataset is sourced from Kaggle's TuSimple lane detection challenge, including 12,816 images split into training and testing sets:

Training: 3626 frames and 3626 lane masks.
Testing: 2782 frames and 2782 lane masks.


2.Algorithm:
The project employs a U-Net model for lane detection

Loss Functions:
- Binary Cross-Entropy: Initially used to compute loss between predicted and actual labels.
- Dice Loss: Introduced to focus on maximizing overlap between predicted and actual labels, enhancing segmentation accuracy.

Results:
- Dice Coefficient: 0.7685
- IoU: 0.4927

3. Future Plans:
Model Refinement: Explore deeper models and enhance robustness.
Deployment: Optimize for real-time use and integrate into larger systems.
