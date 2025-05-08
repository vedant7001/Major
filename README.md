# Breast Cancer Classification using CNN Models

This project implements and compares three different CNN architectures (DenseNet, ResNet, and EfficientNet) for breast cancer classification using ultrasound or mammography images.

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
├── results/               # Results directory (created during training)
│
├── train.py               # Main training script
├── evaluate.py            # Evaluation script
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

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- PyYAML
- opencv-python

You can install the required packages using:

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn tqdm pyyaml opencv-python
```

## Dataset

The code is designed to work with standard image classification datasets where each class has its own directory. For example:

```
data/
├── benign/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── malignant/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── normal/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

You can use public datasets like:
- [BUSI (Breast Ultrasound Images)](https://scholar.cu.edu.eg/Dataset_BUSI.zip)
- [mini-MIAS (Mammographic Image Analysis Society)](http://peipa.essex.ac.uk/info/mias.html)

## Usage

### Training

To train a single model:

```bash
python train.py --model-name densenet --version 121 --data-dir path/to/dataset --batch-size 32 --epochs 30
```

Common options:
- `--model-name`: Model architecture (`densenet`, `resnet`, or `efficientnet`)
- `--version`: Model version (`121`, `169` for DenseNet; `18`, `50`, `101` for ResNet; `b0`, `b3` for EfficientNet)
- `--data-dir`: Path to the dataset
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--no-pretrained`: Disable use of pretrained weights
- `--no-augment`: Disable data augmentation

You can also use a configuration file:

```bash
python train.py --config path/to/config.yaml
```

To train and compare multiple models, uncomment the `train_multiple_models()` line in `train.py` and run:

```bash
python train.py
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model-path path/to/checkpoint.pth --config-path path/to/config.yaml --visualize --gradcam
```

Options:
- `--model-path`: Path to the trained model checkpoint
- `--config-path`: Path to the configuration file
- `--data-dir`: Path to the dataset (if different from config)
- `--results-dir`: Path to save evaluation results
- `--visualize`: Visualize model predictions and feature maps
- `--gradcam`: Generate Grad-CAM visualizations
- `--num-samples`: Number of samples to visualize

## Customization

You can customize various aspects of the project:

### Adding New Models

To add a new model architecture, extend the `models.py` file with a new model class and update the `get_model` function.

### Custom Dataset

If your dataset has a different structure, you can modify the data loading functions in `dataset.py`.

### Training Parameters

You can customize training parameters by modifying the configuration files or by passing command-line arguments to the training script.

## License

This project is available under the MIT License.

## Acknowledgements

This project uses the following open-source libraries:
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
``` 