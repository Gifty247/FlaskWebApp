 Lane Detection Using a U-Net CNN
 
 Table of Contents
1. Dataset
2. Algorithm
3. Setup
4. Future Plans

1. Dataset
The dataset is sourced from Kaggle's TuSimple lane detection challenge, including 12,816 images split into training and testing sets:

Training: 3626 frames and 3626 lane masks.
Testing: 2782 frames and 2782 lane masks.


2.Algorithm
The project employs a U-Net model for lane detection

Loss Functions:
- Binary Cross-Entropy: Initially used to compute loss between predicted and actual labels.
- Dice Loss: Introduced to focus on maximizing overlap between predicted and actual labels, enhancing segmentation accuracy.

Results:
- Dice Coefficient: 0.7685
- IoU: 0.4927

3.Setup
1. Clone the repository:

   ```bash
   git clone https://github.com/Gifty247/FlaskWebApp.git

   
Create a virtual environment:

   ```bash
python -m venv venv


Install dependencies:

  ```bash
pip install -r requirements.txt


Run the application:

  ```bash
python main.py




4. Future Plans
Data Diversity: Include more diverse driving conditions.
Model Refinement: Explore deeper models and enhance robustness.
Deployment: Optimize for real-time use and integrate into larger systems.
2.Algorithm
The project employs a U-Net model for lane detection

Loss Functions:
- Binary Cross-Entropy: Initially used to compute loss between predicted and actual labels.
