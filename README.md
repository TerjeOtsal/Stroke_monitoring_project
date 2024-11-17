# Potential Project Plan 1.01

## Push Changes to Git Repository

1. Navigate to the correct directory in the terminal:

    ```bash
    git add .
    ```

2. Create a new commit:

    ```bash
    git commit -m "Initial commit"
    ```

3. Link the local repository to the GitHub repository:

    ```bash
    git remote add origin https://github.com/TerjeOtsal/Stroke_monitoring_project.git
    ```

4. Push local commits to the branch (replace `tryAgain` with your target branch, or use `main` if it's the default):

    ```bash
    git push -u origin tryAgain
    ```

# Stroke Movement Classification Project

This repository contains the implementation and associated scripts for a machine learning model designed to classify hand movements in post-stroke patients using data from a wearable device. The primary model, `FINAL_MODEL.py`, is a feedforward neural network (FFNN) optimized for this task. Additionally, supplementary scripts are included for data preprocessing, testing, and cleaning.

---

## Main Components

### `FINAL_MODEL.py`
The core script of the project, implementing a feedforward neural network for movement classification.

#### Key Features:
- **Model Architecture**: A FFNN with two hidden layers (64 and 32 neurons) using ReLU activation and L2 regularization.
- **Optimization Techniques**:
  - **Adam Optimizer** for adaptive learning rates.
  - **Gradient Clipping** to stabilize training.
  - **Learning Rate Scheduler** to dynamically adjust the learning rate.
  - **Early Stopping** to prevent overfitting.
- **Classification**: Five movement classes:
  - Hand towards body
  - Hand down
  - Hand outwards
  - Hand upwards
  - Hand forward
- **Data Preprocessing**:
  - Handles NaN and infinite values by replacing them with zeros.
  - Standardizes features using `StandardScaler`.

#### How to Use:
1. Place the labeled datasets (`combined_labeled_stroke_data.csv` and `combined_labeled_stroke_data3.csv`) in the project directory.
2. Run the script:
   ```bash
   python FINAL_MODEL.py

Secondary scripts : model_tester.py test the trained model on new datasets    
Timeremove.py removes timestamp. can easily me modified to remove other columns 
Lstm.py attempt on temporal algorithm Long-short term memory algorithm

