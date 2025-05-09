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

from models.models import get_model
from utils.visualization import visualize_gradcam
from configs.config import load_config
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
    
    # Create model with comprehensive error handling
    try:
        # Disable Streamlit file watcher during model loading
        import warnings
        from streamlit.runtime.scriptrunner import RerunData, RerunException
        
        # Temporarily disable Streamlit's file watcher
        st.runtime.scriptrunner.add_script_run_ctx = lambda *args, **kwargs: None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings('ignore', category=UserWarning, message='.*__path__._path.*')
            
            # Initialize new event loop for PyTorch operations
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                model = get_model(
                    config['model']['model_name'],
                    num_classes=config['model']['num_classes'],
                    pretrained=False,
                    version=config['model']['version']
                )
                
                # Restore original event loop
                loop.close()
                asyncio.set_event_loop(asyncio.new_event_loop())
                
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    model = get_model(
                        config['model']['model_name'],
                        num_classes=config['model']['num_classes'],
                        pretrained=False,
                        version=config['model']['version']
                    )
                    loop.close()
                    asyncio.set_event_loop(asyncio.new_event_loop())
                else:
                    raise
            
    except RuntimeError as e:
        if "Tried to instantiate class" in str(e):
            st.error("Error loading model: PyTorch class initialization failed")
            return None, None
        raise
    except AttributeError as e:
        if "__path__" in str(e):
            st.error("Error loading model: Streamlit watcher conflict with PyTorch")
            return None, None
    except Exception as e:
        st.error(f"Unexpected error loading model: {str(e)}")
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
                    # Google Drive file ID for the model
                    file_id = "1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"
                    output = f"{os.path.dirname(model_path)}/downloaded_model.pth"
                    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
                    checkpoint = torch.load(output, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(device)
                    model.eval()
                    st.success("Model downloaded successfully!")
                    return model, device
                except Exception as download_error:
                    st.error(f"Failed to download model: {str(download_error)}")
                    return None, None
        else:
            st.error(f"Error loading model weights: {str(e)}")
        return None, None
    
    return model, device
    
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
        # Google Drive folder ID for model files
        folder_id = "1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"
        OUTPUT_ZIP = os.path.join(os.getcwd(), "models.zip")
        
        try:
            # Download the entire folder as zip
            gdown.download_folder(id=folder_id, output=OUTPUT_ZIP, quiet=False)
            
            # Extract the zip file
            import zipfile
            with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
                zip_ref.extractall(os.getcwd())
            os.remove(OUTPUT_ZIP)
            
            # Verify extraction
            if os.path.exists(models_dir) and os.listdir(models_dir):
                st.success("Models successfully downloaded and extracted from Google Drive.")
            else:
                st.error("Failed to extract models. Please check the Google Drive folder contents.")
                return
                
        except Exception as e:
            st.error(f"Error downloading models from Google Drive: {str(e)}")
            return
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
        st.info("Please download the model checkpoints from Google Drive")
        
        if st.button("Download missing checkpoints"):
            with st.spinner("Downloading checkpoints..."):
                try:
                    if download_models(models_dir):
                        checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
                        if checkpoint_files:
                            latest_checkpoint = checkpoint_files[-1]
                            model_path = os.path.join(model_dir, latest_checkpoint)
                            st.success("Checkpoints downloaded successfully!")
                            st.rerun()
                        else:
                            st.error("Still no checkpoints found after download")
                    else:
                        st.error("Failed to download checkpoints")
                except Exception as e:
                    st.error(f"Download error: {str(e)}")
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


import gdown

# Google Drive folder ID
folder_id = "1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"

# Check if model files already exist
models_dir = os.path.join("Model", "checkpoints")
if os.path.exists(models_dir) and os.listdir(models_dir):
    st.info("Model files already exist, skipping download.")
else:
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Download with error handling
    try:
        st.info("Downloading model files from Google Drive...")
        gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", 
                            output=models_dir, 
                            quiet=False, 
                            use_cookies=False)
        
        # Verify download
        if not os.listdir(models_dir):
            st.error("Download failed - no files were downloaded")
        else:
            st.success("Model files downloaded successfully")
    except Exception as e:
        st.error(f"Failed to download model files: {str(e)}")