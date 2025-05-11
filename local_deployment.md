# Local Deployment Guide

## Step 1: Clone the repository (if you haven't already)
```bash
git clone https://github.com/vedant7001/Major.git
cd Major
```

## Step 2: Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

## Step 3: Install dependencies
```bash
pip install -r requirements_streamlit.txt
```

## Step 4: Prepare model files
Ensure you have trained model files (.pth) in the appropriate directories:
- `models/checkpoints/colab_densenet121_busi/model_best.pth`
- `models/checkpoints/colab_resnet50_busi/model_best.pth` (optional)
- `models/checkpoints/colab_efficientnetb3_busi/model_best.pth` (optional)

If you don't have model files, the app will attempt to download them when launched.

## Step 5: Run the application
```bash
streamlit run app.py
```

The app should open in your default web browser at http://localhost:8501

## Step 6: Using the application
1. Select a model from the sidebar
2. Upload a breast ultrasound image
3. View the classification results and confidence scores
4. Examine the Grad-CAM visualization

## Troubleshooting
- If you see "streamlit is not recognized as a command", try using `python -m streamlit run app.py`
- If models fail to load, check that the .pth files exist in the correct directories
- For GPU acceleration, ensure you have the correct CUDA version installed 