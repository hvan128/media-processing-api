# Media Processing API Dockerfile
# ================================
# Lightweight container optimized for CPU-only VPS deployment
# 
# Build: docker build -t media-api .
# Run:   docker run -p 8000:8000 -v $(pwd)/data:/data media-api

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    MODEL_PATH=/app

# Install system dependencies
# - ffmpeg: Audio/video processing (runtime)
# - libsndfile1: Audio file reading
# - libgomp: OpenMP library required by CTranslate2 (faster-whisper)
# - curl: Health checks
# Note: librnnoise-dev is not in standard repos, so we use FFmpeg's built-in filters
# with high-pass filtering fallback for speech suppression
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create data directory for temp files and outputs
RUN mkdir -p /data/output /data/jobs

# Download RNNoise model file for arnndn filter
# This model is required for neural network-based speech suppression
RUN curl -L -o /app/rnnoise_model.rnnn \
    "https://github.com/xiph/rnnoise/raw/master/src/rnnoise_model.rnnn" && \
    echo "RNNoise model downloaded"

# Install Python dependencies
# Split into stages for better caching

# Install core dependencies first
RUN pip install "numpy<2.0.0"

# Speech suppression uses FFmpeg's arnndn filter (RNNoise neural network)
# RNNoise model is downloaded above and will be used by arnndn filter
# Falls back to high-pass filtering if arnndn is not available in FFmpeg build

# Install PyAV from pre-built wheel (avoid building from source)
# av>=12 has wheels compatible with manylinux
RUN pip install av>=12.0.0 --only-binary :all:

# Install faster-whisper (skip av since we installed it above)
RUN pip install faster-whisper==1.0.0 --no-deps && \
    pip install ctranslate2 huggingface_hub tokenizers onnxruntime requests

# Install FastAPI and other dependencies
RUN pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 httpx==0.26.0 aiofiles==23.2.1 \
    pydantic==2.5.3 python-dotenv==1.0.0

# Copy application code
COPY *.py .

# Expose API port
EXPOSE 8000

# Health check
# start-period=60s allows time for ML model loading (Whisper)
# Reduced from 90s since RNNoise is much faster to initialize than Spleeter
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# Single worker for sequential processing (memory-constrained environments)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
