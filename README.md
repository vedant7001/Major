# Breast Cancer Classification App

This application uses deep learning to classify breast ultrasound images as Normal, Benign, or Malignant.

## Features

- Multiple model architectures: DenseNet-121, ResNet-50, and EfficientNet-B3
- Real-time prediction with confidence scores
- GradCAM visualization to highlight areas of interest in the image
- User-friendly interface for medical professionals

## How to Use

1. Visit the deployed app at: [Breast Cancer Classification App](https://your-username-major.streamlit.app)
2. Upload a breast ultrasound image
3. Select a model architecture from the sidebar
4. View the prediction results and confidence scores
5. Examine the GradCAM visualization to understand influential regions

## Local Development

### Prerequisites
- Python 3.9+
- pip

### Installation
```bash
git clone https://github.com/vedant7001/Major.git
cd Major
pip install -r requirements.txt
```

### Running Locally
```bash
streamlit run streamlit_app.py
```

## Models

The application uses three pre-trained deep learning models:

1. **DenseNet-121**: Excellent feature extraction with dense connectivity
2. **ResNet-50**: Deep residual learning framework with skip connections
3. **EfficientNet-B3**: Balanced performance and efficiency with compound scaling

Each model was trained on the BUSI dataset (Breast Ultrasound Images) and achieves high accuracy in classification tasks.

## Deployment

This app is deployed on Streamlit Community Cloud. For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## License

[MIT License](LICENSE)

## Contact

For questions or feedback, please contact [your-email].