# Breast Cancer Classification App - Deployment Guide

This guide provides step-by-step instructions for deploying the Breast Cancer Classification application using three different methods:

1. [Local Deployment](#local-deployment)
2. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
3. [Heroku Deployment](#heroku-deployment)

## Local Deployment

### Prerequisites
- Python 3.7+ installed
- Git installed

### Step 1: Clone the repository
```bash
git clone https://github.com/vedant7001/Major.git
cd Major
```

### Step 2: Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements_streamlit.txt
```

### Step 4: Prepare model files
Ensure you have trained model files (.pth) in the appropriate directories:
- `models/checkpoints/colab_densenet121_busi/model_best.pth`
- `models/checkpoints/colab_resnet50_busi/model_best.pth` (optional)
- `models/checkpoints/colab_efficientnetb3_busi/model_best.pth` (optional)

If you don't have model files, the app will attempt to download them when launched.

### Step 5: Run the application
```bash
streamlit run app.py
```
or
```bash
python -m streamlit run app.py
```

The app should open in your default web browser at http://localhost:8501

## Streamlit Cloud Deployment

Streamlit Cloud provides free hosting for your Streamlit applications.

### Step 1: Ensure your code is on GitHub
Make sure your repository is pushed to GitHub with the latest changes:

```bash
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin master
```

### Step 2: Sign up for Streamlit Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign up using your GitHub account
3. Verify your email if required

### Step 3: Deploy your app
1. Click on "New app" button
2. Select your repository (`vedant7001/Major`)
3. Select the branch (usually `master` or `main`)
4. Set the main file path to `app.py`
5. Under "Advanced settings", select requirements_streamlit.txt as your requirements file
6. Click "Deploy"

### Step 4: Handle model files
Since model files are large and not typically stored in GitHub:

1. **Option 1**: Configure the app to download models on first run
   - The app already includes code to download missing models

2. **Option 2**: Use Streamlit secrets to store model download URLs
   - Go to your app settings in Streamlit Cloud
   - Add secrets in TOML format:
     ```toml
     [models]
     densenet_url = "your-cloud-storage-url/model_best.pth"
     ```

### Step 5: Monitor your deployment
1. Wait for the build process to complete
2. Check the logs for any errors
3. Your app will be available at a URL like: `https://vedant7001-major-app-xyz123.streamlit.app`

## Heroku Deployment

Heroku is a platform as a service (PaaS) that enables developers to deploy, manage, and scale applications.

### Step 1: Install the Heroku CLI
Download and install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli):

```bash
# macOS
brew tap heroku/brew && brew install heroku

# Windows
# Download installer from: https://devcenter.heroku.com/articles/heroku-cli

# Ubuntu/Debian
curl https://cli-assets.heroku.com/install-ubuntu.sh | sh
```

### Step 2: Login to Heroku
```bash
heroku login
```

### Step 3: Prepare your application
Ensure you have the following files in your repository:

1. **Procfile** (already exists):
   ```
   web: streamlit run app.py --server.port=$PORT
   ```

2. **requirements.txt** (rename requirements_streamlit.txt):
   ```bash
   cp requirements_streamlit.txt requirements.txt
   ```

3. **runtime.txt** (create this file):
   ```
   python-3.9.16
   ```

### Step 4: Create a Heroku app
```bash
heroku create breast-cancer-classifier-app
```

### Step 5: Set up buildpacks
```bash
heroku buildpacks:set heroku/python
```

### Step 6: Deploy your application
```bash
git add .
git commit -m "Heroku deployment setup"
git push heroku master
```

### Step 7: Configure model file handling
Since Heroku has an ephemeral filesystem and limited slug size (500MB), you need to handle model files carefully:

1. **Option 1**: Configure the app to download models on first run
   - The app already includes code to download missing models
   - Set environment variables for model URLs:
     ```bash
     heroku config:set MODEL_DOWNLOAD_URL="https://your-storage-url/models.zip"
     ```

2. **Option 2**: Use external storage (AWS S3, Google Cloud Storage)
   - Store your models in cloud storage
   - Set environment variables with access credentials:
     ```bash
     heroku config:set AWS_ACCESS_KEY_ID="your-key"
     heroku config:set AWS_SECRET_ACCESS_KEY="your-secret"
     heroku config:set S3_BUCKET_NAME="your-bucket"
     ```

### Step 8: Scale your application
```bash
heroku ps:scale web=1
```

### Step 9: Open your application
```bash
heroku open
```

## Troubleshooting

### Common Issues

#### Streamlit not found
If you see "streamlit is not recognized as a command", try using:
```bash
python -m streamlit run app.py
```

#### Model loading errors
- Check that the .pth files exist in the correct directories
- Ensure your model architecture matches the saved weights

#### Memory issues
- For local deployment: Close other applications to free up memory
- For Streamlit Cloud: Optimize your code to use less memory
- For Heroku: Upgrade to a higher dyno tier

#### Timeout during deployment
- Ensure large files are not included in your git repository
- Configure your app to download models at runtime

#### Package conflicts
- Use a clean virtual environment
- Specify exact package versions in requirements_streamlit.txt

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Heroku Python Support](https://devcenter.heroku.com/articles/python-support)
- [PyTorch Model Deployment Best Practices](https://pytorch.org/tutorials/recipes/recipes/deployment_with_flask.html)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud) 