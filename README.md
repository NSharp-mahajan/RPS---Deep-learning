# RPS CNN Battle

A collaborative deep learning project developed by a team of 4 members, focused on building and evaluating a Convolutional Neural Network (CNN) for classifying Rock, Paper, and Scissors images using TensorFlow and TensorFlow Datasets.

## Project Overview

This project explores the full pipeline of a computer vision task:

- Load the Rock, Paper, Scissors dataset from TensorFlow Datasets
- Inspect dataset distribution and sample images
- Preprocess images for CNN training
- Build a baseline CNN model
- Train and evaluate the model
- Analyze performance using metrics such as accuracy, loss, confusion matrix, and classification reports

The main work is currently documented in the notebook:

- `notebooks/rps_cnn_battle.ipynb`

## Team

This project is being developed by a team of 4 people.

## Current Progress

### 1. Dataset Setup
The project successfully loaded the `rock_paper_scissors` dataset using TensorFlow Datasets.

Key steps completed:
- Downloaded and prepared the dataset
- Loaded train, validation, and test splits
- Verified dataset structure, class names, and image shape

### 2. Data Exploration
The dataset was explored to understand class balance and sample images.

Completed tasks:
- Checked train/test label distributions
- Visualized sample images from the training set
- Confirmed the class labels are:
  - `paper`
  - `rock`
  - `scissors`

### 3. CNN Pipeline Design
A baseline CNN pipeline was designed for the task.

Pipeline stages:
- Resize images to `128 x 128`
- Normalize pixel values from `0-255` to `0-1`
- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- MaxPooling
- GlobalAveragePooling
- Dense layer
- 3-class softmax output

### 4. Baseline Model Training
A baseline CNN model was implemented and trained.

Architecture highlights:
- Input size: `128 x 128 x 3`
- Batch size: `32`
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy

Training setup:
- Training split: 80% of original training split
- Validation split: remaining 20% of original training split
- Test split: kept separate for final evaluation

### 5. Baseline Evaluation
The baseline model was evaluated on the held-out test set.

Key results:
- Training accuracy: `99.50%`
- Validation accuracy: `99.80%`
- Test accuracy: `84.95%`
- Test loss: `0.6262`

### 6. Error Analysis
The model performed very strongly on training and validation data, but accuracy dropped on the test set.

This suggests a possible generalization gap, likely caused by differences in:
- hand pose
- lighting
- background
- image composition
- dataset distribution differences between train/validation and test sets

The project includes:
- confusion matrix
- classification report
- sample predictions with true vs predicted labels

## Repository Structure

```text
RPS-Project/
├── notebooks/
│   └── rps_cnn_battle.ipynb
├── README.md
└── ...
```

## Key Findings

- The dataset is successfully loaded and usable for training.
- The baseline CNN learns the task very well on seen data.
- The trained model generalizes reasonably well, but there is still a visible test-time performance gap.
- The current notebook provides a solid foundation for further experiments such as:
  - data augmentation
  - regularization
  - architecture tuning
  - transfer learning
  - improved evaluation and visualization

## Setup Instructions

### Prerequisites

Install the required Python libraries:

```bash
pip install tensorflow tensorflow-datasets matplotlib numpy scikit-learn
```

### Run the Notebook

Open and execute:

```text
notebooks/rps_cnn_battle.ipynb
```

## Notes

This README reflects the project progress completed so far. The project is currently in the baseline analysis and evaluation phase, with room for further improvements and experiments.


## Conclusion

The project has successfully completed the initial deep learning workflow for a Rock, Paper, Scissors classification task, from dataset loading to baseline training and evaluation. The baseline model demonstrates strong learning capability, while the next phase should focus on improving robustness and reducing the observed train/validation/test performance gap.
