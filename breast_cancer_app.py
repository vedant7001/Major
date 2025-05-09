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

# Initialize event loop for Streamlit with error handling
try:
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
except Exception as e:
    st.error(f"Error initializing event loop: {str(e)}")

# Import custom modules
from models.models import get_model
from configs.config import load_config

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define helper functions
def load_model(model_path, config):
    """Load a trained model with enhanced error handling"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        st.error(f"Error loading model weights: {str(e)}")
        return None, None
    
    return model, device

def preprocess_image(image, config):
    """Preprocess an image for model input"""
    preprocess = transforms.Compose([
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)

def predict(model, image_tensor, device, class_names):
    """Make a prediction on an image"""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    pred_class = class_names[preds.item()]
    pred_prob = probs[0][preds.item()].item()
    all_probs = {class_names[i]: probs[0][i].item() for i in range(len(class_names))}
    
    return pred_class, pred_prob, all_probs

def download_models(models_dir, folder_id="1wh67S5wGO2VnJg4IjNWwQrrj99n7qqy6"):
    """Download models from Google Drive if not available locally"""
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        with st.spinner("Downloading models from Google Drive..."):
            try:
                OUTPUT_ZIP = os.path.join(os.getcwd(), "models.zip")
                gdown.download_folder(id=folder_id, output=OUTPUT_ZIP, quiet=False)
                
                import zipfile
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

# Main application
def main():
    st.title("Breast Cancer Ultrasound Classification")
    st.markdown("""
    This application uses deep learning to classify breast ultrasound images as normal, benign, or malignant.
    Upload an ultrasound image to get a prediction with visual explanations.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Model Settings")
        
        # Check for models directory
        models_dir = os.path.join(os.getcwd(), "models", "checkpoints")
        if not download_models(models_dir):
            return
            
        # Model selection
        available_models = [d for d in os.listdir(models_dir) 
                          if os.path.isdir(os.path.join(models_dir, d))]
        
        if not available_models:
            st.error("No trained models found.")
            return
            
        selected_model = st.selectbox("Select Model", available_models)
        
        # Load model configuration
        model_dir = os.path.join(models_dir, selected_model)
        config_path = os.path.join(model_dir, "config.yaml")
        
        if not os.path.exists(config_path):
            st.error(f"Configuration file not found for {selected_model}")
            return
            
        config = load_config(config_path)
        
        # Find latest checkpoint
        checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        
        if not checkpoint_files:
            st.error(f"No checkpoint files found for {selected_model}")
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
            
        latest_checkpoint = checkpoint_files[-1]
        model_path = os.path.join(model_dir, latest_checkpoint)
        
        # Load model
        model, device = load_model(model_path, config)
        
        if model is None:
            return
            
        # Get class names
        data_dir = config['data']['data_dir']
        class_names = [d for d in sorted(os.listdir(data_dir)) 
                      if os.path.isdir(os.path.join(data_dir, d))]
    
    # Main content area
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader("Choose an ultrasound image", 
                                   type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            image_tensor = preprocess_image(image, config)
            pred_class, pred_prob, all_probs = predict(model, image_tensor, 
                                                     device, class_names)
        
        # Display results
        with col2:
            st.subheader("Prediction Results")
            
            # Confidence indicator
            st.metric(label="Predicted Class", 
                     value=pred_class, 
                     delta=f"{pred_prob:.1%} confidence")
            
            # Probability distribution
            st.subheader("Class Probabilities")
            st.bar_chart({"Probability": all_probs})

# Run the app
if __name__ == "__main__":
    main()