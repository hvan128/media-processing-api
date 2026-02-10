"""
Media Processing API
====================
Self-hosted backend for audio/video processing workflows.

Provides:
- Speech-to-Text (STT) transcription
- Vocal separation (Spleeter)
- Audio-video merging (FFmpeg)

All operations are asynchronous with job polling.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl

from job_manager import job_manager, JobType, JobStatus
from stt_service import load_model as load_whisper_model, process_stt
from separation_service import load_model as load_spleeter_model, process_separation
from merge_service import process_merge
from tts_service import process_tts, save_api_key as save_elevenlabs_api_key
from local_tts_service import (
    load_model as load_valtec_model,
    process_local_tts,
    get_available_speakers as get_tts_speakers
)


# ====================
# Pydantic Models
# ====================

class STTRequest(BaseModel):
    """Request body for Speech-to-Text endpoint"""
    media_url: HttpUrl = Field(..., description="Public URL to audio file (mp3, wav)")
    language: str = Field(default="zh", description="Language code (e.g., 'zh', 'en', 'ja')")
    output: Literal["srt"] = Field(default="srt", description="Output format (only 'srt' supported)")


class SeparateRequest(BaseModel):
    """Request body for Vocal Separation endpoint"""
    media_url: HttpUrl = Field(..., description="Public URL to audio file")


class MergeInput(BaseModel):
    """Single input for merge operation"""
    file_url: HttpUrl = Field(..., description="Public URL to media file")
    type: Literal["video", "audio"] = Field(..., description="Input type: 'video' or 'audio'")
    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Volume multiplier (0.0-2.0, default 1.0)")
    duration_mode: Literal[
        "auto_match_video",
        "always_speed",
        "always_trim_video",
        "keep_original",
    ] = Field(
        default="auto_match_video",
        description=(
            "How to resolve duration mismatch between this audio and the video. "
            "Defaults to 'auto_match_video'."
        ),
    )


class MergeOptions(BaseModel):
    """Options for merge operation"""
    mute_original_audio: bool = Field(default=True, description="Discard original video audio if true")


class MergeRequest(BaseModel):
    """Request body for Audio-Video Merge endpoint"""
    inputs: list[MergeInput] = Field(..., min_length=2, description="List of inputs (exactly 1 video, 1+ audio)")
    options: MergeOptions = Field(default_factory=MergeOptions, description="Merge options")


class TTSRequest(BaseModel):
    """
    Request body for Text-to-Speech endpoint (ElevenLabs - DEPRECATED)
    
    DEPRECATED: Use LocalTTSRequest with POST /local-tts instead.
    
    Example curl:
        curl -X POST http://<server>/tts \\
          -H "Content-Type: application/json" \\
          -d '{
            "text":"Hello world",
            "voice_id":"<voice_id>"
          }'
    """
    text: str = Field(..., description="Text to synthesize into speech")
    voice_id: str = Field(..., description="ElevenLabs voice ID")
    model_id: Optional[str] = Field(default="eleven_monolingual_v1", description="ElevenLabs model ID")
    stability: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Voice stability (0.0-1.0)")
    similarity_boost: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Voice similarity boost (0.0-1.0)")


class LocalTTSRequest(BaseModel):
    """
    Request body for Local Text-to-Speech endpoint (Valtec TTS)
    
    Vietnamese TTS using locally-hosted Valtec model.
    
    Available voices:
    - NF: Northern Female (Miền Bắc - Nữ)
    - SF: Southern Female (Miền Nam - Nữ)
    - NM1: Northern Male 1 (Miền Bắc - Nam)
    - NM2: Northern Male 2 (Miền Bắc - Nam)
    - SM: Southern Male (Miền Nam - Nam)
    
    Example curl:
        # Basic usage:
        curl -X POST http://localhost:8000/local-tts \\
          -H "Content-Type: application/json" \\
          -d '{
            "text": "Xin chào, đây là test TTS local"
          }'
        
        # With options:
        curl -X POST http://localhost:8000/local-tts \\
          -H "Content-Type: application/json" \\
          -d '{
            "text": "Xin chào các bạn",
            "voice": "NF",
            "speed": 1.0
          }'
        
        # Check job:
        curl http://localhost:8000/job/<job_id>
    """
    text: str = Field(..., min_length=1, description="Vietnamese text to synthesize into speech")
    voice: Optional[str] = Field(
        default="NF",
        description="Speaker voice: NF (Northern Female), SF (Southern Female), NM1/NM2 (Northern Male), SM (Southern Male)"
    )
    speed: Optional[float] = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed (0.5-2.0, default 1.0 = normal, < 1.0 = faster, > 1.0 = slower)"
    )


class ElevenLabsConfigRequest(BaseModel):
    """
    Request body for ElevenLabs API key configuration
    
    Example curl:
        curl -X POST http://<server>/config/elevenlabs \\
          -H "Content-Type: application/json" \\
          -d '{"api_key":"YOUR_KEY"}'
    """
    api_key: str = Field(..., description="ElevenLabs API key")


class ConfigResponse(BaseModel):
    """Response for configuration endpoints"""
    status: str = "ok"


class JobCreatedResponse(BaseModel):
    """Response for job creation endpoints"""
    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    """Response for job status endpoint"""
    status: str
    progress: Optional[int] = None
    result: Optional[dict] = None
    error_message: Optional[str] = None


# ====================
# Application Lifecycle
# ====================


def _load_all_models_sync():
    """
    Load ALL ML models synchronously (blocking).
    
    This function is meant to be called in an executor during startup.
    It blocks until all models are loaded, ensuring no race conditions.
    """
    print("Loading Whisper model...")
    load_whisper_model()
    
    print("Loading Spleeter model...")
    load_spleeter_model()
    
    print("Loading Valtec TTS model...")
    load_valtec_model(device="cpu")  # Use CPU for VPS deployment
    
    print("All models loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    Startup:
    - Load ML models FIRST (blocking, in executor)
    - Start job worker and register handlers
    
    Shutdown:
    - Stop job worker gracefully
    
    IMPORTANT: Models are loaded synchronously during startup.
    The API will NOT accept requests until models are ready.
    This eliminates race conditions between startup and job execution.
    """
    import asyncio
    
    # Startup
    print("Starting Media Processing API...")
    
    # Step 1: Load ALL models FIRST (blocking, in executor)
    # This runs in a thread pool to avoid blocking the event loop completely
    # but we AWAIT it to ensure models are ready before accepting requests
    print("Loading ML models (this may take 30-90 seconds)...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_all_models_sync)
    
    # Step 2: Register job handlers (models are now guaranteed to be loaded)
    job_manager.register_handler(JobType.STT, process_stt)
    job_manager.register_handler(JobType.SEPARATE, process_separation)
    job_manager.register_handler(JobType.MERGE, process_merge)
    job_manager.register_handler(JobType.TTS, process_tts)
    job_manager.register_handler(JobType.LOCAL_TTS, process_local_tts)
    
    # Step 3: Start job worker
    await job_manager.start()
    
    print("API server ready! All models loaded.")
    
    yield
    
    # Shutdown
    print("Shutting down...")
    await job_manager.stop()
    print("Goodbye!")


# ====================
# FastAPI Application
# ====================

app = FastAPI(
    title="Media Processing API",
    description="""
    Self-hosted backend for audio/video processing.
    
    ## Features
    - **Speech-to-Text**: Transcribe audio to SRT subtitles
    - **Vocal Separation**: Remove vocals from audio (keep instrumental)
    - **Audio Merge**: Combine multiple audio tracks into video
    - **Local TTS**: Vietnamese text-to-speech using Valtec model (no external API)
    
    ## Usage Pattern
    1. POST to create a job (returns job_id)
    2. GET /job/{job_id} to poll for status
    3. When status is "done", result contains file_url
    """,
    version="1.1.0",
    lifespan=lifespan
)

# Mount static files directory for serving outputs
# This will be available at /static/{filename}
app.mount("/static", StaticFiles(directory=str(job_manager.output_dir)), name="static")


# ====================
# Health Check
# ====================

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns HTTP 200 as long as the FastAPI server is running.
    This endpoint does NOT wait for ML models to load, allowing
    health checks to pass immediately during container startup.
    
    This is critical for CI/CD deployments where:
    - Container health checks must pass within start_period (90-120s)
    - Model loading can take 30-90 seconds
    - Server must be considered "healthy" as soon as it can accept requests
    
    Models are loaded in the background and will be ready when needed.
    """
    # Always return healthy - server is up and can accept requests
    # Model loading happens in background and doesn't block health checks
    return {"status": "healthy", "server": "ready"}


# ====================
# API Endpoints
# ====================

@app.post("/stt", response_model=JobCreatedResponse)
async def create_stt_job(request: STTRequest):
    """
    Create a Speech-to-Text transcription job.
    
    Downloads the audio from the provided URL and transcribes it
    using faster-whisper. Returns SRT formatted subtitles.
    
    - **media_url**: Public URL to audio file (mp3, wav)
    - **language**: Language code (default: "zh")
    - **output**: Output format (only "srt" supported)
    
    Returns job_id for polling via GET /job/{job_id}
    """
    job = await job_manager.create_job(
        JobType.STT,
        {
            "media_url": str(request.media_url),
            "language": request.language,
            "output": request.output
        }
    )
    
    return JobCreatedResponse(job_id=job.job_id, status="pending")


@app.post("/separate", response_model=JobCreatedResponse)
async def create_separation_job(request: SeparateRequest):
    """
    Create a vocal separation job.
    
    Downloads the audio from the provided URL and separates it
    into vocals and no_vocals tracks using Demucs.
    Returns only the no_vocals (instrumental) track.
    
    - **media_url**: Public URL to audio file
    
    Returns job_id for polling via GET /job/{job_id}
    """
    job = await job_manager.create_job(
        JobType.SEPARATE,
        {
            "media_url": str(request.media_url)
        }
    )
    
    return JobCreatedResponse(job_id=job.job_id, status="pending")


@app.post("/merge", response_model=JobCreatedResponse)
async def create_merge_job(request: MergeRequest):
    """
    Create an audio-video merge job.
    
    Downloads all inputs and merges audio tracks into the video.
    Video stream is copied (no re-encoding) for speed.
    Audio tracks are mixed based on volume settings.
    
    - **inputs**: List with exactly 1 video and 1+ audio files
    - **options.mute_original_audio**: Discard original video audio (default: true)
    
    Returns job_id for polling via GET /job/{job_id}
    """
    # Validate input composition
    video_count = sum(1 for i in request.inputs if i.type == "video")
    audio_count = sum(1 for i in request.inputs if i.type == "audio")
    
    if video_count != 1:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly one video input required, got {video_count}"
        )
    
    if audio_count == 0 and request.options.mute_original_audio:
        raise HTTPException(
            status_code=400,
            detail="At least one audio input required when muting original audio"
        )
    
    # Convert to serializable format
    inputs = [
        {
            "file_url": str(i.file_url),
            "type": i.type,
            "volume": i.volume,
            # Duration mode is optional at the API level but defaults
            # to auto_match_video, so we always serialize it explicitly.
            "duration_mode": i.duration_mode,
        }
        for i in request.inputs
    ]
    
    options = {
        "mute_original_audio": request.options.mute_original_audio
    }
    
    job = await job_manager.create_job(
        JobType.MERGE,
        {
            "inputs": inputs,
            "options": options
        }
    )
    
    return JobCreatedResponse(job_id=job.job_id, status="pending")


@app.post("/tts", response_model=JobCreatedResponse)
async def create_tts_job(request: TTSRequest):
    """
    Create a Text-to-Speech job using ElevenLabs.
    
    DEPRECATED: Use POST /local-tts for local Vietnamese TTS instead.
    
    Synthesizes the provided text into speech using the specified voice.
    The API key must be configured first via POST /config/elevenlabs.
    
    - **text**: Text to synthesize into speech
    - **voice_id**: ElevenLabs voice ID
    - **model_id**: ElevenLabs model ID (default: eleven_monolingual_v1)
    - **stability**: Voice stability setting 0.0-1.0 (default: 0.5)
    - **similarity_boost**: Voice similarity boost 0.0-1.0 (default: 0.5)
    
    Returns job_id for polling via GET /job/{job_id}
    
    Example:
        curl -X POST http://<server>/tts \\
          -H "Content-Type: application/json" \\
          -d '{"text":"Hello world","voice_id":"<voice_id>"}'
    """
    job = await job_manager.create_job(
        JobType.TTS,
        {
            "text": request.text,
            "voice_id": request.voice_id,
            "model_id": request.model_id,
            "stability": request.stability,
            "similarity_boost": request.similarity_boost
        }
    )
    
    return JobCreatedResponse(job_id=job.job_id, status="pending")


@app.post("/local-tts", response_model=JobCreatedResponse)
async def create_local_tts_job(request: LocalTTSRequest):
    """
    Create a local Text-to-Speech job using Valtec TTS.
    
    Synthesizes Vietnamese text into speech using locally-hosted Valtec model.
    No external API required - runs entirely on the server.
    
    Available voices:
    - **NF**: Northern Female (Miền Bắc - Nữ) [default]
    - **SF**: Southern Female (Miền Nam - Nữ)
    - **NM1**: Northern Male 1 (Miền Bắc - Nam)
    - **NM2**: Northern Male 2 (Miền Bắc - Nam)
    - **SM**: Southern Male (Miền Nam - Nam)
    
    Parameters:
    - **text**: Vietnamese text to synthesize (required)
    - **voice**: Speaker voice (default: NF)
    - **speed**: Speech speed 0.5-2.0 (default: 1.0)
    
    Returns job_id for polling via GET /job/{job_id}
    
    Example:
        # Create TTS job:
        curl -X POST http://localhost:8000/local-tts \\
          -H "Content-Type: application/json" \\
          -d '{
            "text": "Xin chào, đây là test TTS local"
          }'
        
        # Check job:
        curl http://localhost:8000/job/<job_id>
    """
    # Validate text is not empty
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    job = await job_manager.create_job(
        JobType.LOCAL_TTS,
        {
            "text": request.text,
            "voice": request.voice,
            "speed": request.speed
        }
    )
    
    return JobCreatedResponse(job_id=job.job_id, status="pending")


@app.post("/config/elevenlabs", response_model=ConfigResponse)
async def configure_elevenlabs(request: ElevenLabsConfigRequest):
    """
    Configure ElevenLabs API key.
    
    Saves the API key to disk for use by TTS jobs.
    Creates /data/config directory if it doesn't exist.
    Overwrites any existing key.
    
    - **api_key**: Your ElevenLabs API key
    
    Returns {"status": "ok"} on success.
    
    Example:
        curl -X POST http://<server>/config/elevenlabs \\
          -H "Content-Type: application/json" \\
          -d '{"api_key":"YOUR_KEY"}'
    """
    try:
        save_elevenlabs_api_key(request.api_key)
        return ConfigResponse(status="ok")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save API key: {str(e)}"
        )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a processing job.
    
    Poll this endpoint to track job progress.
    
    Status values:
    - **pending**: Job is queued, waiting to start
    - **running**: Job is being processed (check progress)
    - **done**: Job completed successfully (check result.file_url)
    - **error**: Job failed (check error_message)
    """
    job = await job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = JobStatusResponse(status=job.status.value)
    
    if job.status == JobStatus.RUNNING:
        response.progress = job.progress
    elif job.status == JobStatus.DONE:
        response.result = job.result
    elif job.status == JobStatus.ERROR:
        response.error_message = job.error_message
    
    return response


# ====================
# Error Handlers
# ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# ====================
# Entry Point
# ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        workers=1  # Single worker for sequential processing
    )
