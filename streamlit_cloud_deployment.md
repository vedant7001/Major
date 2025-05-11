# Streamlit Cloud Deployment Guide

Streamlit Cloud provides free hosting for your Streamlit applications. Here's how to deploy your breast cancer classification app:

## Step 1: Ensure your code is on GitHub
Make sure your repository is pushed to GitHub with the latest changes:

```bash
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin master
```

## Step 2: Sign up for Streamlit Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign up using your GitHub account
3. Verify your email if required

## Step 3: Deploy your app
1. Click on "New app" button
2. Select your repository (`vedant7001/Major`)
3. Select the branch (usually `master` or `main`)
4. Set the main file path to `app.py`
5. Click "Deploy"

## Step 4: Configure advanced settings (if needed)
You can configure:
- Python version (3.9+ recommended)
- Package dependencies (should use requirements_streamlit.txt)
- Secrets management (if you need API keys)

## Step 5: Handle model files
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

## Step 6: Monitor your deployment
1. Wait for the build process to complete
2. Check the logs for any errors
3. Your app will be available at a URL like: `https://vedant7001-major-app-xyz123.streamlit.app`

## Step 7: Share your application
- The URL is public and can be shared with anyone
- You can add the URL to your GitHub repository README

## Troubleshooting
- If the app fails to deploy, check the build logs for errors
- Memory issues: Optimize your model loading to reduce memory usage
- Timeout issues: Add caching to expensive operations using `@st.cache_resource` 