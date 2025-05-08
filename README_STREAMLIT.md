# Breast Cancer Ultrasound Classification - Streamlit Deployment

This repository contains a Streamlit web application for breast cancer ultrasound image classification using deep learning models.

## Features

- Upload and classify breast ultrasound images
- Select from multiple trained models
- View prediction confidence scores
- Visualize model attention using Grad-CAM
- Interactive user interface

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Pip package manager

### Installation

1. Clone this repository (if you haven't already):
   ```
   git clone https://github.com/vedant7001/Major.git
   cd Major
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements_streamlit.txt
   ```

### Running the Application

1. Make sure you have trained models in the `models/checkpoints` directory

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. The application will start and open in your default web browser at `http://localhost:8501`

## Usage

1. Select a model from the sidebar
2. Upload a breast ultrasound image using the file uploader
3. View the prediction results and confidence scores
4. Examine the Grad-CAM visualization to see which regions of the image influenced the prediction

## Deployment Options

### Streamlit Cloud

You can deploy this application to Streamlit Cloud for free:

1. Push your code to a GitHub repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Set the main file path to `app.py`

### Heroku

To deploy to Heroku:

1. Create a `Procfile` with the following content:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```

2. Follow the Heroku deployment instructions

## Model Information

This application uses deep learning models trained on breast ultrasound images. The models are trained to classify images into the following categories:

- Normal: Normal breast tissue
- Benign: Benign tumors
- Malignant: Malignant tumors

The models are based on state-of-the-art architectures such as DenseNet, ResNet, and EfficientNet.

## Troubleshooting

- If you encounter memory issues, try reducing the batch size in the configuration
- For GPU acceleration, ensure you have the correct CUDA version installed for your PyTorch version
- If models are not loading, check that the model checkpoint files (.pth) are in the correct directory