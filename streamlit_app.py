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
import asyncio
import nest_asyncio

# Initialize event loop for Streamlit with enhanced error handling
try:
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Prevent Streamlit from inspecting PyTorch modules
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*__path__._path.*')
    
except RuntimeError as e:
    if "no running event loop" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        st.error(f"Error initializing event loop: {str(e)}")
except Exception as e:
    st.error(f"Unexpected error initializing event loop: {str(e)}")

# Import model-related functions
try:
    from models.models import get_model
    from utils.visualization import visualize_gradcam
    from configs.config import load_config
except ImportError:
    st.error("Could not import required modules. Make sure they're available in the deployed environment.")

import sys
import gdown
import zipfile

def download_models(models_dir, folder_id="1sbVgRPYbewte1EdMn7M9qmRqo0dvCJgi"):
    """Download models from Google Drive if not available locally"""
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        with st.spinner("Downloading models from Google Drive..."):
            try:
                OUTPUT_ZIP = os.path.join(os.getcwd(), "models.zip")
                gdown.download_folder(id=folder_id, output=OUTPUT_ZIP, quiet=False)
                
                with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
                    zip_ref.extractall(os.getcwd())
                os.remove(OUTPUT_ZIP)
                
                if os.path.exists(models_dir) and os.listdir(models_dir):
                    st.success("Models downloaded successfully!")
                    return True
                else:
                    st.error("Failed to extract models.")
                    return False
                    
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                return False
    return True

# Set page configuration
st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

# Define functions
def load_model(model_path, config):
    """Load a trained model with enhanced error handling"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = get_model(
            config['model']['model_name'],
            num_classes=config['model']['num_classes'],
            pretrained=False,
            version=config['model']['version']
        )
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None, None
    
    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
    except Exception as e:
        if "No such file or directory" in str(e):
            st.warning(f"Checkpoint file not found at {model_path}")
            if st.button("Download model from Google Drive"):
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    
                    # Google Drive file ID for the model
                    file_id = "1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"
                    output = model_path
                    
                    # Clear any existing download attempts
                    if os.path.exists(output):
                        os.remove(output)
                        
                    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
                    
                    # Verify download was successful
                    if not os.path.exists(output):
                        st.error("Download failed - file not created")
                        return None, None
                        
                    checkpoint = torch.load(output, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(device)
                    model.eval()
                    st.success("Model downloaded and loaded successfully!")
                    return model, device
                except Exception as download_error:
                    st.error(f"Failed to download model: {str(download_error)}")
                    return None, None
        else:
            st.error(f"Error loading model weights: {str(e)}")
        return None, None
    
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
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    # Get class probabilities
    probs = probabilities[0].cpu().numpy()
    
    return predicted_class, probs, class_names[predicted_class]

def generate_gradcam(model, image_tensor, device, class_idx):
    """Generate Grad-CAM visualization for the model's decision"""
    # Add hooks to the model
    activation = {}
    gradients = {}
    
    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]
        return None
        
    def forward_hook(module, input, output):
        activation['value'] = output
        return None
    
    # Get the last convolutional layer
    if hasattr(model, 'features'):
        # DenseNet or similar architectures
        last_conv_layer = None
        for i, m in enumerate(model.features.modules()):
            if isinstance(m, nn.Conv2d):
                last_conv_layer = m
    elif hasattr(model, 'layer4'):
        # ResNet or similar architectures
        last_conv_layer = model.layer4[-1]
    elif hasattr(model, '_modules') and 'features' in model._modules:
        # EfficientNet or similar architectures
        for i, m in enumerate(model.features.modules()):
            if isinstance(m, nn.Conv2d):
                last_conv_layer = m
    else:
        return None
    
    if last_conv_layer is None:
        return None
    
    # Register hooks
    handle_forward = last_conv_layer.register_forward_hook(forward_hook)
    handle_backward = last_conv_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    image_tensor = image_tensor.unsqueeze(0).to(device)
    model.zero_grad()
    output = model(image_tensor)
    
    # Backward pass
    model.zero_grad()
    class_output = output[0, class_idx]
    class_output.backward()
    
    # Get gradients and activations
    gradients_val = gradients['value']
    activations_val = activation['value']
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Calculate weights
    weights = torch.mean(gradients_val, dim=(2, 3), keepdim=True)
    
    # Generate CAM
    cam = torch.sum(weights * activations_val, dim=1, keepdim=True)
    cam = torch.nn.functional.relu(cam)
    cam = cam.squeeze().cpu().detach().numpy()
    
    # Normalize CAM
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    
    return cam

def main():
    """Main application function"""
    st.title("Breast Cancer Classification")
    st.markdown("""
    This application classifies breast ultrasound images into three categories:
    - **Normal**: No abnormality detected
    - **Benign**: Non-cancerous tumor
    - **Malignant**: Cancerous tumor
    
    Upload an ultrasound image to get started.
    """)
    
    # Define class names
    class_names = ["Normal", "Benign", "Malignant"]
    
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Model Architecture",
        ["DenseNet-121", "ResNet-50", "EfficientNet-B3"]
    )
    
    # Map model choice to model parameters
    model_params = {
        "DenseNet-121": {
            "model_name": "densenet",
            "version": "121",
            "checkpoint_path": "models/checkpoints/colab_densenet121_busi/model_best.pth",
            "config_path": "models/checkpoints/colab_densenet121_busi/config.yaml"
        },
        "ResNet-50": {
            "model_name": "resnet",
            "version": "50",
            "checkpoint_path": "models/checkpoints/colab_resnet50_busi/model_best.pth",
            "config_path": "models/checkpoints/colab_resnet50_busi/config.yaml"
        },
        "EfficientNet-B3": {
            "model_name": "efficientnet",
            "version": "b3",
            "checkpoint_path": "models/checkpoints/colab_efficientnetb3_busi/model_best.pth",
            "config_path": "models/checkpoints/colab_efficientnetb3_busi/config.yaml"
        }
    }
    
    # Get current model parameters
    selected_model = model_params[model_choice]
    
    # Load configuration
    try:
        config = load_config(selected_model["config_path"])
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        default_config = {
            "model": {
                "model_name": selected_model["model_name"],
                "version": selected_model["version"],
                "num_classes": 3
            },
            "data": {
                "img_size": 224
            }
        }
        st.warning(f"Using default configuration")
        config = default_config
    
    # Ensure model directory exists
    models_dir = "models/checkpoints"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        
        # Try to download models
        download_success = download_models(models_dir)
        if not download_success:
            st.warning("Could not download models automatically. You may need to upload a model checkpoint.")
    
    # Load model
    model, device = load_model(selected_model["checkpoint_path"], config)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display the image
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image and make prediction
        if model is not None:
            # Preprocess image
            image_tensor = preprocess_image(image, config)
            
            # Make prediction
            predicted_class, probabilities, class_label = predict(model, image_tensor, device, class_names)
            
            # Display prediction
            with col2:
                st.subheader("Prediction Results")
                
                # Create a colored box for the prediction
                color_map = {
                    "Normal": "green",
                    "Benign": "orange",
                    "Malignant": "red"
                }
                
                st.markdown(
                    f"""
                    <div style="background-color: {color_map[class_label]}; padding: 10px; border-radius: 5px;">
                        <h3 style="color: white; text-align: center;">{class_label}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display probabilities
                st.subheader("Class Probabilities")
                for i, class_name in enumerate(class_names):
                    st.progress(float(probabilities[i]))
                    st.write(f"{class_name}: {probabilities[i]*100:.2f}%")
                
                # Generate and display Grad-CAM
                cam = generate_gradcam(model, image_tensor, device, predicted_class)
                if cam is not None:
                    st.subheader("Grad-CAM Visualization")
                    
                    # Convert PIL Image to numpy array for visualization
                    img_array = np.array(image.resize((224, 224)))
                    
                    # Resize CAM to match image dimensions
                    cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
                    
                    # Create heatmap
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Combine original image with heatmap
                    alpha = 0.4
                    superimposed_img = heatmap * alpha + img_array
                    superimposed_img = np.uint8(superimposed_img / superimposed_img.max() * 255)
                    
                    # Display the Grad-CAM visualization
                    st.image(superimposed_img, caption='Grad-CAM Visualization', use_column_width=True)
                    st.markdown("""
                    **Grad-CAM Explanation**: The highlighted areas (in red/yellow) show the regions 
                    that most influenced the model's prediction.
                    """)
        else:
            st.error("Model could not be loaded. Please check the model path and configuration.")
    else:
        # Display example images when no file is uploaded
        st.info("Please upload an ultrasound image to get started, or see the examples below.")
        
        # You can add example images here
        st.subheader("Example Classifications")
        st.markdown("""
        - **Normal**: No abnormality detected
        - **Benign**: Non-cancerous tumor
        - **Malignant**: Cancerous tumor
        """)

if __name__ == "__main__":
    main() 