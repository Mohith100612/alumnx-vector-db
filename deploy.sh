#!/bin/bash
set -e

# Arguments
BRANCH="${1:-dev}"
PORT="${2:-8001}"
APP_NAME="${3:-alumnx-vector-db-dev}"
REPO_DIR="/home/ubuntu/$APP_NAME"

echo "Deploying $APP_NAME (Branch: $BRANCH) on Port: $PORT..."

# Initialize directory if not exists
if [ ! -d "$REPO_DIR" ]; then
    echo "Creating directory $REPO_DIR and cloning repo..."
    git clone https://github.com/alumnx-ai-labs/alumnx-vector-db.git "$REPO_DIR"
fi

# Navigate to repo
cd "$REPO_DIR"

# Discard any local changes so checkout never fails
git fetch origin
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
git clean -fd

# Install dependencies
if command -v uv > /dev/null; then
    uv sync --frozen
else
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
fi

# Reload or start with PM2
if pm2 describe "$APP_NAME" > /dev/null 2>&1; then
    echo "Reloading $APP_NAME..."
    pm2 reload "$APP_NAME"
else
    echo "Starting $APP_NAME for the first time..."
    pm2 start "uv run uvicorn main:app --host 0.0.0.0 --port $PORT" --name "$APP_NAME"
fi

pm2 save
echo "Deployment completed successfully!"
