# Media Processing API Dockerfile
# ================================
# Container with support for:
# - Speech-to-Text (faster-whisper)
# - Speech Suppression (RNNoise)
# - Audio Merge (FFmpeg)
# - Local TTS (Valtec Vietnamese TTS)
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
# - git: For cloning valtec-tts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    git \
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

# ====================
# Install Python dependencies
# ====================

# Install core dependencies first
RUN pip install "numpy<2.0.0"

# Install PyTorch CPU version (smaller footprint for VPS)
RUN pip install torch==2.5.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install PyAV from pre-built wheel (avoid building from source)
RUN pip install av>=12.0.0 --only-binary :all:

# Install faster-whisper (skip av since we installed it above)
RUN pip install faster-whisper==1.0.0 --no-deps && \
    pip install ctranslate2 huggingface_hub tokenizers onnxruntime requests

# Install FastAPI and server dependencies
RUN pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 httpx==0.26.0 aiofiles==23.2.1 \
    pydantic==2.5.3 python-dotenv==1.0.0

# ====================
# Install Valtec TTS dependencies
# ====================

# Audio processing
RUN pip install scipy>=1.10.0 soundfile>=0.12.0 librosa>=0.9.0 tqdm>=4.60.0

# Text processing for Vietnamese
RUN pip install Unidecode>=1.3.0 num2words>=0.5.10 inflect>=6.0.0 \
    viphoneme>=3.0.0 underthesea>=8.0.0 vinorm>=2.0.0

# Phonemization dependencies
RUN pip install cn2an>=0.5.20 jieba>=0.42.0 pypinyin>=0.44.0 jamo>=0.4.1 \
    gruut>=2.4.0 g2p-en>=2.1.0 anyascii>=0.3.0 eng-to-ipa>=0.0.2

# ====================
# Clone and install Valtec TTS
# ====================

# Clone valtec-tts repository
RUN git clone --depth 1 https://github.com/tronghieuit/valtec-tts.git /app/valtec-tts

# Install valtec-tts as editable package (allows imports)
# Note: We use --no-deps to avoid reinstalling torch and other deps
RUN cd /app/valtec-tts && pip install -e . --no-deps

# Pre-download Valtec TTS model during build (optional but recommended)
# This avoids download during first request
# Uncomment if you want to pre-cache the model:
# RUN python -c "from valtec_tts import TTS; TTS(device='cpu')"

# ====================
# Copy application code
# ====================

COPY *.py .

# Expose API port
EXPOSE 8000

# Health check
# start-period=120s allows time for ML model loading (Whisper + Valtec TTS)
# Increased from 60s to account for Valtec TTS model download on first run
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# Single worker for sequential processing (memory-constrained environments)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
