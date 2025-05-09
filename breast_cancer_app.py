import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import gdown
import zipfile
import traceback
import shutil

# Set page config
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model download function
def download_models(models_dir, source_type="gdrive", source_path=None):
    """
    Download or load models from different sources
    
    Args:
        models_dir (str): Directory to save models
        source_type (str): Source type - 'gdrive', 'local', or 'url'
        source_path (str): Path/URL to models (required for 'local' and 'url' types)
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    with st.spinner(f"Loading models from {source_type}..."):
        try:
            if source_type == "gdrive":
                # Default Google Drive folder ID
                folder_id = "1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"
                
                # Download the folder directly to models_dir
                gdown.download_folder(id=folder_id, output=models_dir, quiet=False)
                
                # Check if downloads were successful
                if not os.listdir(models_dir):
                    st.error("No files were downloaded from Google Drive")
                    return False
            
            elif source_type == "local" and source_path:
                if os.path.isfile(source_path) and source_path.endswith('.zip'):
                    with zipfile.ZipFile(source_path, 'r') as zip_ref:
                        zip_ref.extractall(models_dir)
                elif os.path.isdir(source_path):
                    shutil.copytree(source_path, models_dir, dirs_exist_ok=True)
            
            elif source_type == "url" and source_path:
                import requests
                from io import BytesIO
                
                response = requests.get(source_path)
                with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(models_dir)
            
            # Verify models were downloaded
            if os.path.exists(models_dir) and os.listdir(models_dir):
                st.success("Models loaded successfully")
                return True
            else:
                st.error("Failed to load models from specified source")
                return False
                
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            st.code(traceback.format_exc())
            return False

# Function to ensure image has 3 channels
def ensure_rgb(image):
    """Convert image to RGB if it's grayscale"""
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def main():
    st.title("Breast Cancer Ultrasound Classification")
    
    # Define models directory
    models_dir = os.path.join(os.getcwd(), "models")
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    
    # Always show image upload first (for better UX)
    st.sidebar.header("📷 Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an ultrasound image...", type=["jpg", "png", "jpeg"], key="image_uploader")
    
    # Add a divider
    st.sidebar.markdown("---")
    st.sidebar.header("Model Selection")
    
    # Create models directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Check for models and provide download button
    download_button = st.sidebar.button("Download/Update Models")
    if download_button:
        download_success = download_models(models_dir)
        if download_success:
            # Move models to checkpoints directory if needed
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path) and item != "checkpoints":
                    try:
                        target_path = os.path.join(checkpoints_dir, item)
                        if os.path.exists(target_path):
                            shutil.rmtree(target_path)
                        shutil.move(item_path, target_path)
                        st.sidebar.success(f"Moved {item} to checkpoints")
                    except Exception as e:
                        st.sidebar.error(f"Error moving {item}: {str(e)}")
        
            st.experimental_rerun()
    
    # Model selection
    try:
        available_models = [d for d in os.listdir(checkpoints_dir) 
                        if os.path.isdir(os.path.join(checkpoints_dir, d))]
    except Exception as e:
        st.error(f"Error listing checkpoint directories: {str(e)}")
        st.code(traceback.format_exc())
        available_models = []
    
    if not available_models:
        st.warning("No trained models found. Please download models using the sidebar button.")
        
        # Display uploaded image even if no models are available
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image (No models available for prediction)', use_column_width=True)
                st.info("No models found. Cannot make predictions.")
            except Exception as e:
                st.error(f"Error opening image: {str(e)}")
        return
    
    # Display model selection options
    selected_model = st.sidebar.selectbox("Available Models", available_models)
    st.sidebar.info(f"Selected model: {selected_model}")
    
    # Load model and make predictions
    model_path = os.path.join(checkpoints_dir, selected_model, 'model.pth')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image (Model file not found)', use_column_width=True)
            except Exception as e:
                st.error(f"Error opening image: {str(e)}")
        return
    
    # Load model with enhanced error handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None  # Initialize model variable
    
    try:
        st.info(f"Loading model from {model_path}...")
        try:
            # First try loading just the state dict (preferred method)
            from torchvision import models
            # This is a placeholder - you should use your actual model architecture
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes
            
            # Load the state dict
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success("Loaded model using state_dict approach")
        except Exception as state_dict_error:
            st.warning(f"Couldn't load state_dict directly. Trying full model load...")
            # Fallback to loading the entire model object
            model = torch.load(model_path, map_location=device)
            st.success("Loaded full model object")
        
        if model is not None:
            model.to(device)
            model.eval()
            st.success(f"Model {selected_model} loaded successfully!")
        else:
            st.error("Failed to initialize model")
            return
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.code(traceback.format_exc())
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image (Error loading model)', use_column_width=True)
            except Exception as img_error:
                st.error(f"Error opening image: {str(img_error)}")
        return
    
    # Process prediction if both model and image are available
    if uploaded_file is not None:
        try:
            # Open and display image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Ultrasound Image', use_column_width=True)
            
            # Ensure image is RGB
            image = ensure_rgb(image)
            
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Create tensor and move to device
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                probs = probabilities.cpu().numpy()
                
                # Get class prediction
                _, predicted = torch.max(outputs.data, 1)
                
            # Display results
            classes = ['Normal', 'Benign', 'Malignant']
            prediction = classes[predicted.item()]
            
            # Create a more detailed results display
            st.subheader("Classification Results")
            st.write(f"**Prediction: {prediction}**")
            
            # Display probabilities
            st.write("### Class Probabilities")
            for i, (cls, prob) in enumerate(zip(classes, probs)):
                st.progress(float(prob))
                st.write(f"{cls}: {prob:.2%}")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.code(traceback.format_exc())
            st.write("Please ensure the uploaded file is a valid image.")
    else:
        st.info("📤 Please upload an ultrasound image using the sidebar to get predictions.")

if __name__ == "__main__":
    main()