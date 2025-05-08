import os
import sys
import argparse
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'torch', 'torchvision', 'numpy', 'matplotlib',
        'PIL', 'cv2', 'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'yaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")
    
    return missing_packages

def check_model_files():
    """Check if model files exist"""
    models_dir = os.path.join(os.getcwd(), "models", "checkpoints")
    
    if not os.path.exists(models_dir):
        print(f"✗ Models directory not found: {models_dir}")
        return False
    
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    if not model_dirs:
        print(f"✗ No model directories found in {models_dir}")
        return False
    
    for model_dir in model_dirs:
        full_model_dir = os.path.join(models_dir, model_dir)
        config_path = os.path.join(full_model_dir, "config.yaml")
        checkpoint_files = [f for f in os.listdir(full_model_dir) if f.endswith(".pth")]
        
        if not os.path.exists(config_path):
            print(f"✗ Configuration file not found for model {model_dir}")
            return False
        
        if not checkpoint_files:
            print(f"✗ No checkpoint files found for model {model_dir}")
            return False
        
        print(f"✓ Model {model_dir} is valid (has config and checkpoint)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test Streamlit app setup')
    parser.add_argument('--install', action='store_true', help='Install missing dependencies')
    args = parser.parse_args()
    
    print("\n===== Testing Streamlit App Setup =====")
    
    # Check dependencies
    print("\nChecking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n✗ Missing packages: {', '.join(missing_packages)}")
        if args.install:
            print("\nInstalling missing packages...")
            for package in missing_packages:
                os.system(f"{sys.executable} -m pip install {package}")
            print("\nPlease run this script again to verify installation.")
        else:
            print("\nPlease install the missing packages using:")
            print("pip install -r requirements_streamlit.txt")
            print("\nOr run this script with --install flag:")
            print("python test_streamlit_app.py --install")
        return
    
    print("\n✓ All required packages are installed!")
    
    # Check model files
    print("\nChecking model files...")
    models_ok = check_model_files()
    
    if not models_ok:
        print("\n✗ Model files check failed!")
        print("Please make sure you have trained models in the models/checkpoints directory.")
        return
    
    print("\n✓ Model files check passed!")
    
    # Check app.py
    app_path = os.path.join(os.getcwd(), "app.py")
    if not os.path.exists(app_path):
        print(f"\n✗ app.py not found at {app_path}")
        return
    
    print(f"\n✓ app.py found at {app_path}")
    
    # All checks passed
    print("\n===== All checks passed! =====")
    print("You can now run the Streamlit app using:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()