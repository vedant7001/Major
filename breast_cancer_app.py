import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import gdown
import zipfile

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
    
    if not os.listdir(models_dir):
        with st.spinner(f"Loading models from {source_type}..."):
            try:
                if source_type == "gdrive":
                    # Default Google Drive folder ID
                    folder_id = "1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"
                    OUTPUT_ZIP = os.path.join(os.getcwd(), "models.zip")
                    
                    # Download the folder
                    gdown.download_folder(id=folder_id, output=models_dir, quiet=False)
                    
                    # Check if files were downloaded directly or as a zip
                    if os.path.exists(OUTPUT_ZIP):
                        with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
                            zip_ref.extractall(models_dir)
                        os.remove(OUTPUT_ZIP)
                
                elif source_type == "local" and source_path:
                    if os.path.isfile(source_path) and source_path.endswith('.zip'):
                        with zipfile.ZipFile(source_path, 'r') as zip_ref:
                            zip_ref.extractall(models_dir)
                    elif os.path.isdir(source_path):
                        import shutil
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
                return False
    return True

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
    
    # Check for models
    if not download_models(models_dir):
        st.error("Failed to download models")
        return
        
    # Check for checkpoints directory
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
        st.warning("Checkpoints directory created but no models found. Please refresh after model download.")
        return
        
    # Model selection
    available_models = [d for d in os.listdir(checkpoints_dir) 
                      if os.path.isdir(os.path.join(checkpoints_dir, d))]
    
    if not available_models:
        st.error("No trained models found in checkpoints directory")
        return
        
    selected_model = st.sidebar.selectbox("Select Model", available_models)
    
    # Load model and make predictions
    model_path = os.path.join(checkpoints_dir, selected_model, 'model.pth')
    
    # Load model with enhanced error handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Load model properly with state_dict
        if os.path.exists(model_path):
            # Note: This is a safer way to load models, but requires knowledge of the architecture
            # For the fixed version, we'll use a try-except to handle both loading methods
            try:
                # First try loading just the state dict (preferred method)
                from torchvision import models
                # This is a placeholder - you should use your actual model architecture
                model = models.resnet50(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes
                model.load_state_dict(torch.load(model_path, map_location=device))
            except:
                # Fallback to loading the entire model object
                model = torch.load(model_path, map_location=device)
                
            model.to(device)
            model.eval()
            st.success(f"Successfully loaded {selected_model} model")
        else:
            st.warning(f"Model file not found at {model_path}")
            if st.sidebar.button("Download missing model"):
                try:
                    # Use the same Google Drive folder ID as download_models
                    folder_id = "1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"
                    
                    # Download directly to models directory
                    gdown.download_folder(id=folder_id, output=models_dir, quiet=False)
                    
                    st.success("Model downloaded successfully! Please refresh the page.")
                    return
                except Exception as download_error:
                    st.error(f"Failed to download model: {str(download_error)}")
                    return
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    # Image upload and prediction
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an ultrasound image...", type=["jpg", "png", "jpeg"])
    
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
            st.write("Please ensure the uploaded file is a valid image.")

if __name__ == "__main__":
    main()