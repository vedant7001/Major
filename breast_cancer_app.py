import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import asyncio
import nest_asyncio
import gdown
import zipfile

# Initialize event loop
nest_asyncio.apply()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

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
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        with st.spinner(f"Loading models from {source_type}..."):
            try:
                if source_type == "gdrive":
                    # Default Google Drive folder ID
                    folder_id = "1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"
                    OUTPUT_ZIP = os.path.join(os.getcwd(), "models.zip")
                    gdown.download_folder(id=folder_id, output=OUTPUT_ZIP, quiet=False)
                    
                    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
                        zip_ref.extractall(os.getcwd())
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
                
                if os.path.exists(models_dir) and os.listdir(models_dir):
                    return True
                else:
                    st.error("Failed to load models from specified source")
                    return False
                    
            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")
                return False
    return True

def main():
    st.title("Breast Cancer Ultrasound Classification")
    
    # Check for models
    models_dir = os.path.join(os.getcwd(), "models", "checkpoints")
    if not download_models(models_dir):
        st.error("Failed to download models")
        return
        
    # Model selection
    available_models = [d for d in os.listdir(models_dir) 
                      if os.path.isdir(os.path.join(models_dir, d))]
    
    if not available_models:
        st.error("No trained models found")
        return
        
    selected_model = st.sidebar.selectbox("Select Model", available_models)
    
    # Load model and make predictions
    model_path = os.path.join(models_dir, selected_model, 'model.pth')
    
    # Load model with enhanced error handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        st.success(f"Successfully loaded {selected_model} model")
    except Exception as e:
        if "No such file or directory" in str(e):
            st.warning(f"Model file not found at {model_path}")
            if st.sidebar.button("Download missing model"):
                try:
                    # Use the same Google Drive folder ID as download_models
                    folder_id = "1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"
                    OUTPUT_ZIP = os.path.join(os.getcwd(), "models.zip")
                    gdown.download_folder(id=folder_id, output=OUTPUT_ZIP, quiet=False)
                    
                    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
                        zip_ref.extractall(os.getcwd())
                    os.remove(OUTPUT_ZIP)
                    
                    st.success("Model downloaded successfully! Please refresh the page.")
                    return
                except Exception as download_error:
                    st.error(f"Failed to download model: {str(download_error)}")
                    return
        else:
            st.error(f"Failed to load model: {str(e)}")
        return
    
    # Image upload and prediction
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an ultrasound image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Ultrasound Image', use_column_width=True)
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
            # Display results
            classes = ['Normal', 'Benign', 'Malignant']
            st.success(f"Prediction: {classes[predicted.item()]}")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()