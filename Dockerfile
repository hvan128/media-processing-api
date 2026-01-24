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
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# - ffmpeg: Audio/video processing (runtime)
# - libavformat-dev, libavcodec-dev, etc.: FFmpeg dev libraries for PyAV
# - libsndfile1: Audio file reading (for Demucs)
# - libgomp: OpenMP library required by CTranslate2 (faster-whisper)
# - git: Required by some Python packages
# - curl: Health checks
# - pkg-config: Required for building PyAV
# - gcc, python3-dev: Build tools for compiling native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libsndfile1 \
    libgomp1 \
    git \
    curl \
    pkg-config \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create data directory for temp files and outputs
RUN mkdir -p /data/output /data/jobs

# Install Python dependencies
# Split into stages for better caching:
# 1. Install NumPy first (required by PyTorch)
# 2. Install PyTorch CPU-only (large, rarely changes)
# 3. Install PyAV (requires FFmpeg dev libs)
# 4. Install other dependencies

# Install NumPy (required by PyTorch, install first to avoid compatibility issues)
# Pin to version compatible with PyTorch 2.1.2
RUN pip install "numpy<1.25.0"

# Install PyTorch CPU-only (saves ~1GB vs full torch)
RUN pip install torch==2.1.2+cpu torchaudio==2.1.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install PyAV (required by faster-whisper)
RUN pip install av

# Copy requirements and install remaining dependencies
COPY requirements.txt .

# Install remaining dependencies (skip torch as it's already installed)
RUN pip install --no-deps faster-whisper==1.0.0 && \
    pip install demucs==4.0.1 && \
    pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 httpx==0.26.0 aiofiles==23.2.1 \
    pydantic==2.5.3 python-dotenv==1.0.0

# Install remaining faster-whisper dependencies (excluding av which is already installed)
RUN pip install ctranslate2 huggingface_hub tokenizers onnxruntime requests

# Pre-download ML models during build (optional, reduces first-run latency)
# Uncomment these lines to include models in the image:
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
# RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

# Copy application code
COPY *.py .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# Single worker for sequential processing (memory-constrained environments)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
