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
    CUDA_VISIBLE_DEVICES=-1

# Install system dependencies
# - ffmpeg: Audio/video processing (runtime)
# - libsndfile1: Audio file reading (for Spleeter)
# - libgomp: OpenMP library required by CTranslate2 (faster-whisper)
# - curl: Health checks
# Note: No FFmpeg dev libs needed - using pre-built PyAV wheel
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

# Install Python dependencies
# Split into stages for better caching

# Install core dependencies first
RUN pip install "numpy<2.0.0"

# Install Spleeter (uses TensorFlow, CPU-only by default)
# Note: We use Spleeter as a library, not CLI, to avoid typer compatibility issues
RUN pip install spleeter==2.4.0

# Install PyAV from pre-built wheel (avoid building from source)
# av>=12 has wheels compatible with manylinux
RUN pip install av>=12.0.0 --only-binary :all:

# Install faster-whisper (skip av since we installed it above)
RUN pip install faster-whisper==1.0.0 --no-deps && \
    pip install ctranslate2 huggingface_hub tokenizers onnxruntime

# Install FastAPI and other dependencies
RUN pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 httpx==0.26.0 aiofiles==23.2.1 \
    pydantic==2.5.3 python-dotenv==1.0.0

# Pre-download Spleeter model during build (reduces first-run latency)
# Note: GitHub releases return 302 redirects that keras downloader doesn't handle
# We download manually with curl which follows redirects properly
ENV MODEL_PATH=/root/pretrained_models
RUN mkdir -p /root/pretrained_models/2stems && \
    curl -L -o /tmp/2stems.tar.gz \
      "https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz" && \
    tar -xzf /tmp/2stems.tar.gz -C /root/pretrained_models/2stems && \
    rm /tmp/2stems.tar.gz && \
    echo "Spleeter model downloaded"

# Copy application code
COPY *.py .

# Expose API port
EXPOSE 8000

# Health check
# start-period=90s allows time for ML model loading (Whisper)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# Single worker for sequential processing (memory-constrained environments)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
