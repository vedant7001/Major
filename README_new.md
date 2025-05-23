# Breast Cancer Classification Project

This project implements and compares three different CNN architectures (DenseNet, ResNet, and EfficientNet) for breast cancer classification using ultrasound or mammography images.

## Google Colab Integration

A Jupyter notebook (`notebook.ipynb`) is included to run this project directly on Google Colab. The notebook:
- Clones this GitHub repository
- Sets up the necessary environment
- Downloads and prepares the breast cancer dataset
- Trains and evaluates models
- Visualizes results with confusion matrices and GradCAM

### Running on Colab

1. Open Google Colab (https://colab.research.google.com/)
2. Go to File > Open notebook
3. Select the GitHub tab
4. Enter the repository URL: https://github.com/vedant7001/Major
5. Select the notebook.ipynb file

## Project Structure

```
breast_cancer_classification/
│
├── configs/               # Configuration files
│   └── config.py          # Configuration utilities
│
├── data/                  # Data loading and preprocessing
│   └── dataset.py         # Dataset classes and data loaders
│
├── models/                # Model definitions
│   └── models.py          # CNN model architectures
│
├── utils/                 # Utility functions
│   ├── train_utils.py     # Training and evaluation utilities
│   └── visualization.py   # Visualization utilities
│
├── train.py               # Main training script
├── evaluate.py            # Evaluation script
├── notebook.ipynb         # Jupyter notebook for Colab
└── README.md              # Project documentation
```

## Features

- Implementation of three CNN architectures:
  - DenseNet121 (or DenseNet169)
  - ResNet50 (or ResNet18/101)
  - EfficientNetB0 (or EfficientNetB3)
- Data preprocessing and augmentation
- Training with early stopping and learning rate scheduling
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, AUC
- Visualization of results:
  - Training/validation curves
  - Confusion matrices
  - ROC curves
  - Feature maps and Grad-CAM visualizations
- Model comparison tools

## Dataset

The code supports standard image classification datasets where each class has its own directory.
Recommended datasets:
- [BUSI (Breast Ultrasound Images)](https://scholar.cu.edu.eg/Dataset_BUSI.zip)
- [mini-MIAS (Mammographic Image Analysis Society)](http://peipa.essex.ac.uk/info/mias.html)

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- scikit-learn
- matplotlib
- seaborn
- tqdm
- PyYAML
- opencv-python 