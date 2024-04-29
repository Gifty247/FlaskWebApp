 Lane Detection Using a U-Net CNN
 
 Table of Contents
1. Dataset
2. Algorithm
3. Setup
4. Future Plans

Dataset
The dataset is sourced from Kaggle's TuSimple lane detection challenge, including 12,816 images split into training and testing sets:

Training: 3626 frames and 3626 lane masks.
Testing: 2782 frames and 2782 lane masks.


Algorithm
A U-Net model is used for lane detection:


Loss Functions:
- Binary Cross-Entropy: For initial loss computation.
- Dice Loss: To maximize overlap between predicted and actual labels.

Results:
- Dice Coefficient: 0.7685
- IoU: 0.4927

Setup
1. Clone the repository:

   ```bash
   git clone https://github.com/Gifty247/FlaskWebApp.git
   
Create a virtual environment:

python -m venv venv

Install dependencies:

pip install -r requirements.txt

Run the application:

python main.py

Future Plans
Data Diversity: Include more diverse driving conditions.
Model Refinement: Explore deeper models and enhance robustness.
Deployment: Optimize for real-time use and integrate into larger systems.
