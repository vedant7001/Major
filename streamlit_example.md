# Using the Breast Cancer Classification App

## Step 1: Select a Model
From the sidebar, select one of the trained models available in your system.

## Step 2: Upload an Image
Upload a breast ultrasound image using the file uploader.

## Step 3: View Results
The app will display:
- The prediction (Normal, Benign, or Malignant)
- Confidence score for the prediction
- Bar chart showing probabilities for all classes
- Grad-CAM visualization highlighting important regions

## Example Interface

```
+---------------------------------------+
|  Breast Cancer Ultrasound Classification  |
+---------------------------------------+
|                                       |
| [Sidebar]  |  [Main Content]         |
| Model:     |  Upload an image:       |
| DenseNet   |  [Upload Button]        |
|            |                         |
|            |  [Image Preview]        |
|            |                         |
|            |  Prediction: Benign     |
|            |  Confidence: 92%        |
|            |                         |
|            |  [Class Probabilities]  |
|            |  Normal:    5%          |
|            |  Benign:    92%         |
|            |  Malignant: 3%          |
|            |                         |
|            |  [Grad-CAM Visualization]|
|            |                         |
+---------------------------------------+
```

## Troubleshooting

- If you encounter an error about missing models, make sure you have trained models in the `models/checkpoints` directory
- If the app crashes, check that all dependencies are installed with `pip install -r requirements_streamlit.txt`
- For more help, run the test script: `python test_streamlit_app.py`