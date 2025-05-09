import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2
from torchvision import transforms

from models.models import get_model
from utils.visualization import visualize_gradcam
from configs.config import load_config
import sys
import gdown

# Set page configuration
st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

# Define functions
def load_model(model_path, config):
    """Load a trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(
        config['model']['model_name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        version=config['model']['version']
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

def preprocess_image(image, config):
    """Preprocess an image for model input"""
    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply preprocessing
    image_tensor = preprocess(image)
    return image_tensor

def predict(model, image_tensor, device, class_names):
    """Make a prediction on an image"""
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Get prediction and confidence
    pred_class = class_names[preds.item()]
    pred_prob = probs[0][preds.item()].item()
    
    # Get all class probabilities
    all_probs = {class_names[i]: probs[0][i].item() for i in range(len(class_names))}
    
    return pred_class, pred_prob, all_probs

def generate_gradcam(model, image_tensor, device, class_idx):
    """Generate Grad-CAM visualization"""
    # Get target layer for Grad-CAM
    if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
        target_layer = model.features.denseblock4
    elif hasattr(model, 'base_model') and len(model.base_model) > 7:
        target_layer = model.base_model[7]  # ResNet's last residual block
    elif hasattr(model, 'features') and len(model.features) > 0:
        target_layer = model.features[-1]
    else:
        st.warning("Grad-CAM not supported for this model architecture")
        return None
    
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_()
    
    # Forward pass
    output = model(image_tensor)
    
    # Clear gradients
    model.zero_grad()
    
    # Backward pass with target class
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)
    
    # Get gradients and activations
    gradients = None
    activations = None
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    # Forward and backward pass
    output = model(image_tensor)
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Calculate Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]).cpu()
    activations = activations[0].cpu()
    
    for i in range(activations.shape[0]):
        activations[i] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=0).numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Convert image to numpy for visualization
    orig_img = image_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    orig_img = orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    orig_img = np.clip(orig_img, 0, 1)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Combine original image and heatmap
    superimposed_img = orig_img * 0.6 + heatmap * 0.4
    superimposed_img = np.clip(superimposed_img, 0, 1)
    
    return superimposed_img

# Main app
def main():
    st.title("Breast Cancer Ultrasound Classification")
    st.write("""
    This application uses deep learning to classify breast ultrasound images as normal, benign, or malignant.
    Upload an ultrasound image to get a prediction.
    """)
    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    # Find available models
    models_dir = os.path.join(os.getcwd(), "models", "checkpoints")
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        st.warning("No models found locally. Attempting to download from Google Drive...")
        # --- CONFIGURE THIS ---
        # Set your Google Drive file ID or URL here
        GDRIVE_FILE_ID = st.secrets["GDRIVE_FILE_ID"] if "GDRIVE_FILE_ID" in st.secrets else "YOUR_FILE_ID_HERE"
        OUTPUT_ZIP = os.path.join(os.getcwd(), "models.zip")
        if GDRIVE_FILE_ID != "YOUR_FILE_ID_HERE":
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", OUTPUT_ZIP, quiet=False)
            import zipfile
            with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
                zip_ref.extractall(os.getcwd())
            os.remove(OUTPUT_ZIP)
            st.success("Models downloaded and extracted.")
        else:
            st.error("Google Drive file ID not set. Please set GDRIVE_FILE_ID in Streamlit secrets or code.")
        # Refresh models_dir after download
        if not os.path.exists(models_dir) or not os.listdir(models_dir):
            st.error("Model download failed or no models found after extraction.")
            return
    available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not available_models:
        st.error("No trained models found. Please train a model first or ensure checkpoint files are present in the correct directory. If you recently downloaded models, verify extraction was successful and the .pth checkpoint files exist in each model folder.")
        return
    
    # Model selection
    selected_model = st.sidebar.selectbox("Select Model", available_models)
    
    # Load model configuration
    model_dir = os.path.join(models_dir, selected_model)
    config_path = os.path.join(model_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found for model {selected_model}")
        return
    
    config = load_config(config_path)
    
    # Find model checkpoint
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    
    if not checkpoint_files:
        st.error(f"No checkpoint files found for model {selected_model}")
        return
    
    # Use the latest checkpoint
    latest_checkpoint = checkpoint_files[-1]
    model_path = os.path.join(model_dir, latest_checkpoint)
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model(model_path, config)
    
    # Get class names
    data_dir = config['data']['data_dir']
    class_names = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Image upload
    st.subheader("Upload an ultrasound image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image and make prediction
        with st.spinner("Analyzing image..."):
            image_tensor = preprocess_image(image, config)
            pred_class, pred_prob, all_probs = predict(model, image_tensor, device, class_names)
        
        # Display prediction
        with col2:
            st.subheader("Prediction")
            st.write(f"**Class:** {pred_class}")
            st.write(f"**Confidence:** {pred_prob:.2%}")
            
            # Display probability bar chart
            st.subheader("Class Probabilities")
            probs_df = {"Class": list(all_probs.keys()), "Probability": list(all_probs.values())}
            st.bar_chart(probs_df, x="Class", y="Probability")
        
        # Generate and display Grad-CAM
        st.subheader("Grad-CAM Visualization")
        st.write("Highlighting regions that influenced the prediction")
        
        # Get class index for prediction
        pred_idx = class_names.index(pred_class)
        
        # Generate Grad-CAM
        gradcam_img = generate_gradcam(model, image_tensor, device, pred_idx)
        
        if gradcam_img is not None:
            st.image(gradcam_img, caption=f"Grad-CAM for class: {pred_class}", use_column_width=True)

# Run the app
if __name__ == "__main__":
    main()