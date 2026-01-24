"""
Media Processing API
====================
Self-hosted backend for audio/video processing workflows.

Provides:
- Speech-to-Text (STT) transcription
- Vocal separation (Demucs)
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
from separation_service import load_model as load_demucs_model, process_separation
from merge_service import process_merge


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


class MergeOptions(BaseModel):
    """Options for merge operation"""
    mute_original_audio: bool = Field(default=True, description="Discard original video audio if true")


class MergeRequest(BaseModel):
    """Request body for Audio-Video Merge endpoint"""
    inputs: list[MergeInput] = Field(..., min_length=2, description="List of inputs (exactly 1 video, 1+ audio)")
    options: MergeOptions = Field(default_factory=MergeOptions, description="Merge options")


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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    Startup:
    - Load ML models into memory
    - Start job worker
    - Register job handlers
    
    Shutdown:
    - Stop job worker gracefully
    """
    # Startup
    print("Starting Media Processing API...")
    
    # Load ML models (do this first as it takes time)
    print("Loading Whisper model...")
    load_whisper_model()
    
    print("Loading Demucs model...")
    load_demucs_model()
    
    # Register job handlers
    job_manager.register_handler(JobType.STT, process_stt)
    job_manager.register_handler(JobType.SEPARATE, process_separation)
    job_manager.register_handler(JobType.MERGE, process_merge)
    
    # Start job worker
    await job_manager.start()
    
    print("API ready!")
    
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
    
    ## Usage Pattern
    1. POST to create a job (returns job_id)
    2. GET /job/{job_id} to poll for status
    3. When status is "done", result contains file_url
    """,
    version="1.0.0",
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
    """Basic health check endpoint"""
    return {"status": "healthy"}


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
            "volume": i.volume
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
