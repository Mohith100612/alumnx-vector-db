#!/bin/bash
set -e

REPO_DIR="/home/ubuntu/alumnx-vector-db"
APP_NAME="alumnx-vector-db"

# Navigate to repo
cd $REPO_DIR

# Pull latest changes
git pull origin main

# Standardize environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Install dependencies (using uv if available, otherwise pip)
if command -v uv > /dev/null; then
    uv sync
else
    ./venv/bin/pip install -r requirements.txt
fi

# Reload or Start with PM2
pm2 describe $APP_NAME > /dev/null
if [ $? -eq 0 ]; then
    echo "Reloading $APP_NAME..."
    pm2 reload $APP_NAME
else
    echo "Starting $APP_NAME for the first time..."
    pm2 start "venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001" --name $APP_NAME
fi

# Save pm2 state
pm2 save

echo "Deployment completed successfully!"
