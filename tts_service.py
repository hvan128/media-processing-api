"""
Text-to-Speech Service Module
=============================
Handles text-to-speech synthesis using ElevenLabs API.

Design Decisions:
- Reads API key from disk file (/data/config/elevenlabs_api_key.txt)
- Uses httpx for HTTP calls to ElevenLabs
- Runs network-heavy work in executor to keep event loop responsive
- Saves MP3 output to static directory for serving

Usage (curl examples):

# Configure ElevenLabs API key:
# curl -X POST http://<server>/config/elevenlabs \
#   -H "Content-Type: application/json" \
#   -d '{"api_key":"YOUR_KEY"}'

# Create TTS job:
# curl -X POST http://<server>/tts \
#   -H "Content-Type: application/json" \
#   -d '{
#     "text":"Hello world",
#     "voice_id":"<voice_id>"
#   }'

# Poll result:
# curl http://<server>/job/<job_id>
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import httpx

from job_manager import Job, JobManager

# Set up logging
logger = logging.getLogger(__name__)

# ElevenLabs API configuration
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_REQUEST_TIMEOUT = 120.0  # 2 minutes for TTS generation

# Config file paths
CONFIG_DIR = Path("/data/config")
ELEVENLABS_API_KEY_FILE = CONFIG_DIR / "elevenlabs_api_key.txt"


def _read_api_key() -> str:
    """
    Read ElevenLabs API key from config file.
    
    Returns:
        API key string
        
    Raises:
        RuntimeError: If file doesn't exist or is empty
    """
    if not ELEVENLABS_API_KEY_FILE.exists():
        raise RuntimeError(
            f"ElevenLabs API key file not found at {ELEVENLABS_API_KEY_FILE}. "
            "Please configure the API key using POST /config/elevenlabs endpoint."
        )
    
    api_key = ELEVENLABS_API_KEY_FILE.read_text().strip()
    
    if not api_key:
        raise RuntimeError(
            f"ElevenLabs API key file is empty at {ELEVENLABS_API_KEY_FILE}. "
            "Please configure a valid API key using POST /config/elevenlabs endpoint."
        )
    
    return api_key


def save_api_key(api_key: str) -> None:
    """
    Save ElevenLabs API key to config file.
    
    Creates the config directory if it doesn't exist.
    Overwrites existing key.
    
    Args:
        api_key: The API key to save
    """
    # Create config directory if needed
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write key to file
    ELEVENLABS_API_KEY_FILE.write_text(api_key.strip())
    
    logger.info(f"ElevenLabs API key saved to {ELEVENLABS_API_KEY_FILE}")


def _call_elevenlabs_sync(
    text: str,
    voice_id: str,
    model_id: str,
    stability: float,
    similarity_boost: float,
    output_path: Path
) -> None:
    """
    Call ElevenLabs TTS API (SYNCHRONOUS, blocking).
    
    Sends text to ElevenLabs and saves the returned MP3 audio to disk.
    
    Args:
        text: Text to synthesize
        voice_id: ElevenLabs voice ID
        model_id: ElevenLabs model ID
        stability: Voice stability setting (0.0-1.0)
        similarity_boost: Voice similarity boost setting (0.0-1.0)
        output_path: Path to save the MP3 output
        
    Raises:
        RuntimeError: On API key issues or API errors
    """
    logger.info(f"Calling ElevenLabs TTS API for voice {voice_id}")
    
    # Read API key from file
    api_key = _read_api_key()
    
    # Build request
    url = f"{ELEVENLABS_API_URL}/{voice_id}"
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }
    
    # Make synchronous HTTP call
    with httpx.Client(timeout=httpx.Timeout(ELEVENLABS_REQUEST_TIMEOUT)) as client:
        response = client.post(url, headers=headers, json=body)
        
        # Check for errors
        if response.status_code != 200:
            error_detail = response.text[:500] if response.text else "No error details"
            raise RuntimeError(
                f"ElevenLabs API error (HTTP {response.status_code}): {error_detail}"
            )
        
        # Save MP3 to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
    
    logger.info(f"TTS audio saved to: {output_path}")


def _process_tts_sync(job: Job, manager: JobManager) -> dict:
    """
    Process a TTS job (SYNCHRONOUS, blocking).
    
    Args:
        job: Job instance with params:
            - text: Text to synthesize
            - voice_id: ElevenLabs voice ID
            - model_id: Optional model ID (default: eleven_monolingual_v1)
            - stability: Optional stability (default: 0.5)
            - similarity_boost: Optional similarity boost (default: 0.5)
        manager: JobManager instance
        
    Returns:
        dict with 'file_url' pointing to the generated MP3 file
    """
    logger.info(f"Processing TTS job {job.job_id}")
    
    params = job.params
    text = params["text"]
    voice_id = params["voice_id"]
    model_id = params.get("model_id", "eleven_monolingual_v1")
    stability = params.get("stability", 0.5)
    similarity_boost = params.get("similarity_boost", 0.5)
    
    # Output path in static directory
    output_filename = f"{job.job_id}.mp3"
    output_path = manager.output_dir / output_filename
    
    # Call ElevenLabs API
    _call_elevenlabs_sync(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        stability=stability,
        similarity_boost=similarity_boost,
        output_path=output_path
    )
    
    # Verify output exists
    if not output_path.exists():
        raise RuntimeError(f"Failed to generate TTS output at {output_path}")
    
    logger.info(f"TTS job {job.job_id} completed successfully")
    
    # Cleanup working directory (nothing to clean for TTS, but call for consistency)
    manager.cleanup_job_work_dir(job)
    
    return {
        "file_url": f"/static/{output_filename}"
    }


async def process_tts(job: Job, manager: JobManager) -> dict:
    """
    Process a text-to-speech job using ElevenLabs API (ASYNC wrapper).
    
    This function schedules blocking HTTP work in an executor to keep the event loop responsive.
    
    Steps:
    1. Read API key from config file
    2. Call ElevenLabs TTS API (in executor)
    3. Save MP3 output to static directory
    
    Args:
        job: Job instance with params:
            - text: Text to synthesize
            - voice_id: ElevenLabs voice ID  
            - model_id: Optional model ID (default: eleven_monolingual_v1)
            - stability: Optional stability (default: 0.5)
            - similarity_boost: Optional similarity boost (default: 0.5)
        manager: JobManager for progress updates
        
    Returns:
        dict with 'file_url' pointing to the generated MP3 file
    """
    logger.info(f"Starting TTS job {job.job_id}")
    
    # Update progress - starting
    await manager.update_progress(job.job_id, 10)
    
    # Schedule blocking ElevenLabs call in executor
    loop = asyncio.get_running_loop()
    
    await manager.update_progress(job.job_id, 20)
    
    logger.info("Scheduling ElevenLabs TTS call in executor")
    
    # Run blocking work in executor (default ThreadPoolExecutor)
    result = await loop.run_in_executor(
        None,
        _process_tts_sync,
        job,
        manager
    )
    
    # Update progress after executor completes
    await manager.update_progress(job.job_id, 100)
    
    logger.info(f"TTS job {job.job_id} completed")
    
    return result
