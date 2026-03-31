# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (FIXES your git error)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies + DVC with S3 support
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir dvc[s3]

# Copy your entire project (code + .dvc files, NOT data)
COPY . .

# Ensure directories exist (optional safety)
RUN mkdir -p /app/data /app/model

# Default command (can be overridden by KFP)
CMD ["python", "train.py"]