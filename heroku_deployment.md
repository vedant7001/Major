# Heroku Deployment Guide

Heroku is a platform as a service (PaaS) that enables developers to deploy, manage, and scale applications. Here's how to deploy your breast cancer classification app to Heroku:

## Step 1: Install the Heroku CLI
Download and install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli):

```bash
# macOS
brew tap heroku/brew && brew install heroku

# Windows
# Download installer from: https://devcenter.heroku.com/articles/heroku-cli

# Ubuntu/Debian
curl https://cli-assets.heroku.com/install-ubuntu.sh | sh
```

## Step 2: Login to Heroku
```bash
heroku login
```

## Step 3: Prepare your application
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

## Step 4: Create a Heroku app
```bash
heroku create breast-cancer-classifier-app
```

## Step 5: Set up buildpacks
```bash
heroku buildpacks:set heroku/python
```

## Step 6: Deploy your application
```bash
git add .
git commit -m "Heroku deployment setup"
git push heroku master
```

## Step 7: Configure model file handling
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

## Step 8: Scale your application
```bash
heroku ps:scale web=1
```

## Step 9: Open your application
```bash
heroku open
```

## Troubleshooting
- **Memory issues**: Upgrade to a higher dyno tier if you encounter memory limits
- **Timeout during build**: Ensure model files are downloaded at runtime, not during build
- **Application crashes**: Check logs with `heroku logs --tail`
- **Slug size too large**: Make sure large files (models) are not in your git repository

## Additional Resources
- [Heroku Python Support](https://devcenter.heroku.com/articles/python-support)
- [Streamlit on Heroku](https://towardsdatascience.com/quickly-build-and-deploy-an-application-with-streamlit-988ca08c7e83) 