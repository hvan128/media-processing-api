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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl

from job_manager import job_manager, JobType, JobStatus
from stt_service import load_model as load_whisper_model, process_stt
from separation_service import load_model as load_demucs_model, process_separation
from merge_service import process_merge
from tts_service import process_tts, save_api_key as save_elevenlabs_api_key
from revid_tts_service import process_revid_tts, save_api_keys as save_revid_api_keys
from eco88labs_tts_service import process_eco88labs_tts, save_api_keys as save_eco88labs_api_keys, get_voices as get_eco88labs_voices, get_any_active_key as get_eco88labs_key


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


class SubtitleSegment(BaseModel):
    """A single subtitle cue with timing and text."""
    start: str = Field(..., description="Start timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm)")
    end: str = Field(..., description="End timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm)")
    text: str = Field(..., min_length=1, description="Subtitle text content")


class RevidTTSRequest(BaseModel):
    """
    Request body for Revid SRT-to-Speech endpoint.

    Uses Revid API (srt-to-speech/merge). Poll GET /job/{job_id} for status.
    When done, result.file_url is the audio (e.g. /static/{job_id}.mp3).

    Example:
        curl -X POST http://localhost:8000/revid_tts \\
          -H "Content-Type: application/json" \\
          -d '{
            "subtitles": [
              {"start": "00:00:00,000", "end": "00:00:02,500", "text": "Đoạn 1"},
              {"start": "00:00:02,500", "end": "00:00:05,000", "text": "Đoạn 2"}
            ],
            "voice_id": 1000,
            "speed": 1.0,
            "language": "vi-VN",
            "add_silence": 0.5
          }'
    """
    subtitles: list[SubtitleSegment] = Field(
        ..., min_length=1, description="Ordered list of subtitle segments (start, end, text)"
    )
    voice_id: int = Field(default=1000, description="Revid Voice ID from Voice Library")
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.5, description="Speech speed (0.5–2.5)")
    language: Optional[str] = Field(default="vi-VN", description="Language code (e.g. vi-VN, en-US)")
    add_silence: Optional[float] = Field(default=0.5, ge=0.0, description="Silence between segments (seconds)")


class Eco88LabsTTSRequest(BaseModel):
    """
    Request body for Eco88Labs Text-to-Speech endpoint.

    Uses Eco88Labs API (/tts-infer). Poll GET /job/{job_id} for status.
    When done, result.file_url is the audio (e.g. /static/{job_id}.wav).

    Example:
        curl -X POST http://localhost:8000/eco88labs_tts \\
          -H "Content-Type: application/json" \\
          -d '{
            "gen_text": "Xin chào, đây là một thử nghiệm.",
            "name_character": "female_voice_01",
            "output_format": "mp3",
            "speed": "1.0"
          }'
    """
    gen_text: str = Field(..., min_length=1, max_length=20000, description="Văn bản cần chuyển đổi (tối đa 20,000 ký tự)")
    name_character: str = Field(..., description="Tên giọng đọc (lấy từ GET /eco88labs/voices)")
    output_format: Optional[str] = Field(default="wav", description="Định dạng đầu ra: wav hoặc mp3 (mặc định: wav)")
    sample_rate: Optional[float] = Field(default=None, description="Tần số lấy mẫu (ví dụ: 24.0, 48.0)")
    speed: Optional[str] = Field(default="1.0", description="Tốc độ đọc (ví dụ: '0.9', '1.1')")
    seed: Optional[int] = Field(default=None, description="Seed để tái tạo kết quả")


class Eco88LabsConfigRequest(BaseModel):
    """
    Request body for Eco88Labs API keys configuration (rotation on 402).

    Example:
        curl -X POST http://<server>/config/eco88labs \\
          -H "Content-Type: application/json" \\
          -d '{"api_keys": ["key1", "key2"]}'
    """
    api_keys: list[str] = Field(..., min_length=1, description="Danh sách Eco88Labs API keys (xoay vòng khi hết balance)")


class RevidConfigRequest(BaseModel):
    """
    Request body for Revid API keys configuration (rotation on 402).

    Example:
        curl -X POST http://<server>/config/revid \\
          -H "Content-Type: application/json" \\
          -d '{"api_keys": ["key1", "key2"]}'
    """
    api_keys: list[str] = Field(..., min_length=1, description="List of Revid API keys (used in order when one runs out of credits)")


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
    
    print("Loading Demucs model...")
    load_demucs_model()
    
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
    job_manager.register_handler(JobType.REVID_TTS, process_revid_tts)
    job_manager.register_handler(JobType.ECO88LABS_TTS, process_eco88labs_tts)

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
    - **Revid TTS**: SRT-to-Speech via Revid API with API key rotation on 402 (insufficient credits)

    ## Usage Pattern
    1. POST to create a job (returns job_id)
    2. GET /job/{job_id} to poll for status
    3. When status is "done", result contains file_url
    """,
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for serving outputs
app.mount("/static", StaticFiles(directory=str(job_manager.output_dir)), name="static")
# Also serve outputs at /output for separation endpoint compatibility
app.mount("/output", StaticFiles(directory=str(job_manager.output_dir)), name="output")


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
    Create a vocal separation job using Demucs (htdemucs).
    
    Downloads the audio and separates it into **vocal** and **instrumental**
    tracks.  Processing runs locally (no third-party API).
    
    When the job is done, the result contains:
    - ``vocal_url``: path to the isolated vocals WAV
    - ``instrument_url``: path to the instrumental WAV
    
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


@app.post("/revid_tts", response_model=JobCreatedResponse)
async def create_revid_tts_job(request: RevidTTSRequest):
    """
    Create a Revid SRT-to-Speech job.

    Sends subtitles to Revid API (srt-to-speech/merge), polls until completed,
    then saves audio to output. Configure API keys via POST /config/revid.
    On 402 (insufficient credits), the service rotates to the next configured key.

    - **subtitles**: List of {start, end, text} (same format as /srt-to-speech)
    - **voice_id**: Revid Voice ID from Voice Library (integer, default 1000)
    - **speed**: 0.5–2.5 (default 1.0)
    - **language**: e.g. vi-VN, en-US (default vi-VN)
    - **add_silence**: Seconds of silence between segments (default 0.5)

    Returns job_id for polling via GET /job/{job_id}. Result has file_url (e.g. /static/{job_id}.mp3).
    """
    subtitles_raw = [
        {"start": s.start, "end": s.end, "text": s.text}
        for s in request.subtitles
    ]
    job = await job_manager.create_job(
        JobType.REVID_TTS,
        {
            "subtitles": subtitles_raw,
            "voice_id": request.voice_id,
            "speed": request.speed,
            "language": request.language,
            "add_silence": request.add_silence,
        },
    )
    return JobCreatedResponse(job_id=job.job_id, status="pending")


@app.post("/eco88labs_tts", response_model=JobCreatedResponse)
async def create_eco88labs_tts_job(request: Eco88LabsTTSRequest):
    """
    Create an Eco88Labs Text-to-Speech job.

    Sends gen_text to Eco88Labs API (/tts-infer), polls /tts/status/<task_id> until completed,
    then saves audio to output. Configure API keys via POST /config/eco88labs.
    On 402 (INSUFFICIENT_TOKEN_BALANCE), the service rotates to the next configured key.

    - **gen_text**: Văn bản cần đọc (tối đa 20,000 ký tự)
    - **name_character**: Tên giọng đọc (lấy từ GET /eco88labs/voices)
    - **output_format**: wav hoặc mp3 (mặc định: wav)
    - **sample_rate**: Tần số lấy mẫu (tùy chọn)
    - **speed**: Tốc độ đọc, ví dụ "0.9", "1.1" (mặc định: "1.0")
    - **seed**: Seed để tái tạo kết quả (tùy chọn)

    Returns job_id for polling via GET /job/{job_id}. Result has file_url (e.g. /static/{job_id}.wav).
    """
    job = await job_manager.create_job(
        JobType.ECO88LABS_TTS,
        {
            "gen_text": request.gen_text,
            "name_character": request.name_character,
            "output_format": request.output_format,
            "sample_rate": request.sample_rate,
            "speed": request.speed,
            "seed": request.seed,
        },
    )
    return JobCreatedResponse(job_id=job.job_id, status="pending")


@app.get("/eco88labs/voices")
async def list_eco88labs_voices():
    """
    Lấy danh sách giọng đọc từ Eco88Labs.

    Sử dụng API key đã cấu hình (POST /config/eco88labs).
    Trả về danh sách voices với name, gender, language, area, emotion.
    """
    try:
        voices = get_eco88labs_voices(get_eco88labs_key())
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/eco88labs", response_model=ConfigResponse)
async def configure_eco88labs(request: Eco88LabsConfigRequest):
    """
    Configure Eco88Labs API keys for rotation.

    When a key returns 402 (INSUFFICIENT_TOKEN_BALANCE), the next key is used.
    Save one key per line in config; order is preserved.
    """
    try:
        save_eco88labs_api_keys(request.api_keys)
        return ConfigResponse(status="ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save Eco88Labs keys: {str(e)}")


@app.post("/config/revid", response_model=ConfigResponse)
async def configure_revid(request: RevidConfigRequest):
    """
    Configure Revid API keys for rotation.

    When a key returns 402 (insufficient credits), the next key is used.
    Save one key per line in config; order is preserved.
    """
    try:
        save_revid_api_keys(request.api_keys)
        return ConfigResponse(status="ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save Revid keys: {str(e)}")


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
    - **done**: Job completed successfully (check result)
    - **error**: Job failed (check error_message)
    
    All fields are always present in the response (null when not applicable).
    """
    job = await job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        status=job.status.value,
        progress=job.progress,
        result=job.result,
        error_message=job.error_message,
    )


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
