## Base image
#FROM python:3.11-slim
#
## Set working directory
#WORKDIR /app
#
## Install system dependencies (FIXES your git error)
#RUN apt-get update && apt-get install -y \
#    git \
#    curl \
#    && rm -rf /var/lib/apt/lists/*
#
## Copy dependency file first (for better layer caching)
#COPY requirements.txt .
#
## Install Python dependencies + DVC with S3 support
#RUN pip install --no-cache-dir -r requirements.txt && \
#    pip install --no-cache-dir dvc[s3]
#
## Copy your entire project (code + .dvc files, NOT data)
#COPY . .
#
## Ensure directories exist (optional safety)
#RUN mkdir -p /app/data /app/model
#
## Default command (can be overridden by KFP)
#CMD ["python", "train.py"]

# ─────────────────────────────────────────────
# Churn Prediction Training Image
# Build: docker build -t crysis307/churn-train:v13 .
# Push:  docker push crysis307/churn-train:v13
#
# What changed from v12:
#   - Removed torch (not used in training — RandomForest only)
#   - Removed dvc, fastapi, uvicorn, pydantic (not needed in container)
#   - Removed hardcoded dvc[s3] pip install
#   - Image should be ~500-700MB instead of 5.71GB
# ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project — preserving folder structure
COPY training/ /app/training/

# Sanity check — fail build if train.py is missing
RUN python -c "import importlib.util; assert importlib.util.find_spec('sklearn'), 'sklearn missing'"

CMD ["python", "/app/training/train.py"]