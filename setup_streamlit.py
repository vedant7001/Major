import os
import sys
import subprocess
import argparse

def install_requirements():
    """Install required packages from requirements_streamlit.txt"""
    req_file = os.path.join(os.getcwd(), "requirements_streamlit.txt")
    
    if not os.path.exists(req_file):
        print(f"Error: {req_file} not found!")
        return False
    
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {str(e)}")
        return False

def check_model_directory():
    """Check if model directory exists and has trained models"""
    models_dir = os.path.join(os.getcwd(), "models", "checkpoints")
    
    if not os.path.exists(models_dir):
        print(f"Warning: Models directory not found: {models_dir}")
        print("Creating models directory...")
        os.makedirs(models_dir, exist_ok=True)
        print("✓ Models directory created")
        print("Note: You need to add trained models to this directory before running the app")
        return False
    
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    if not model_dirs:
        print(f"Warning: No model directories found in {models_dir}")
        print("Note: You need to add trained models to this directory before running the app")
        return False
    
    valid_models = 0
    for model_dir in model_dirs:
        full_model_dir = os.path.join(models_dir, model_dir)
        config_path = os.path.join(full_model_dir, "config.yaml")
        checkpoint_files = [f for f in os.listdir(full_model_dir) if f.endswith(".pth")]
        
        if os.path.exists(config_path) and checkpoint_files:
            valid_models += 1
            print(f"✓ Found valid model: {model_dir}")
    
    if valid_models > 0:
        print(f"✓ Found {valid_models} valid model(s)")
        return True
    else:
        print("Warning: No valid models found (need both config.yaml and .pth files)")
        print("Note: You need to add trained models before running the app")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup Streamlit app for breast cancer classification')
    parser.add_argument('--skip-install', action='store_true', help='Skip package installation')
    args = parser.parse_args()
    
    print("===== Setting up Breast Cancer Classification Streamlit App =====")
    
    # Install requirements
    if not args.skip_install:
        if not install_requirements():
            print("Failed to install requirements. Please install them manually:")
            print("pip install -r requirements_streamlit.txt")
    else:
        print("Skipping package installation...")
    
    # Check model directory
    check_model_directory()
    
    # Check if app.py exists
    app_path = os.path.join(os.getcwd(), "app.py")
    if not os.path.exists(app_path):
        print(f"Error: app.py not found at {app_path}")
        return
    
    print(f"✓ Found app.py at {app_path}")
    
    # Setup complete
    print("\n===== Setup Complete =====")
    print("You can now run the Streamlit app using:")
    print("streamlit run app.py")
    print("\nOr use the deployment helper:")
    print("streamlit run deploy_streamlit.py")

if __name__ == "__main__":
    main()