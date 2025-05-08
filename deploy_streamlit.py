import streamlit as st
import subprocess
import os
import sys

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import torch
        import torchvision
        import numpy
        import matplotlib
        import PIL
        import cv2
        st.success("✅ All required packages are installed!")
        return True
    except ImportError as e:
        st.error(f"❌ Missing package: {str(e)}")
        st.info("Please install all required packages using: pip install -r requirements_streamlit.txt")
        return False

def main():
    st.set_page_config(page_title="Streamlit Deployment Helper", layout="wide")
    
    st.title("Breast Cancer Classification - Deployment Helper")
    st.write("""
    This utility helps you deploy the Breast Cancer Classification app to Streamlit Cloud or run it locally.
    """)
    
    # Check if requirements are installed
    requirements_ok = check_requirements()
    
    if requirements_ok:
        st.subheader("Deployment Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Local Deployment")
            st.write("Run the app on your local machine:")
            if st.button("Run Locally"):
                try:
                    st.info("Starting Streamlit app... Please wait.")
                    # Get the current directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    app_path = os.path.join(current_dir, "app.py")
                    
                    # Check if app.py exists
                    if not os.path.exists(app_path):
                        st.error("app.py not found in the current directory!")
                    else:
                        # Run the streamlit app in a subprocess
                        process = subprocess.Popen(
                            [sys.executable, "-m", "streamlit", "run", app_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        # Display the URL
                        st.success("✅ App is running!")
                        st.markdown("**Open the app at: [http://localhost:8501](http://localhost:8501)**")
                        
                        # Show log output
                        st.subheader("App Log Output:")
                        log_output = st.empty()
                        
                        # Stream the output
                        for line in process.stdout:
                            log_output.text(line)
                except Exception as e:
                    st.error(f"Error starting the app: {str(e)}")
        
        with col2:
            st.markdown("### Streamlit Cloud Deployment")
            st.write("""
            To deploy to Streamlit Cloud:
            
            1. Push your code to GitHub
            2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
            3. Create a new app and connect to your GitHub repo
            4. Set the main file path to `app.py`
            """)
            
            st.markdown("### Heroku Deployment")
            st.write("""
            To deploy to Heroku:
            
            1. Create a `Procfile` with: `web: streamlit run app.py --server.port=$PORT`
            2. Push to Heroku using the Heroku CLI
            """)
    
    st.markdown("---")
    st.markdown("""
    For more detailed instructions, please refer to the `README_STREAMLIT.md` file.
    """)

if __name__ == "__main__":
    main()