"""
Local Text-to-Speech Service Module
===================================
Handles text-to-speech synthesis using Valtec TTS (local Vietnamese TTS model).

Design Decisions:
- Model loaded once at startup and reused for all requests
- Uses valtec-tts with auto-download from Hugging Face
- Runs blocking inference in executor to keep event loop responsive
- Outputs MP3 (converts from WAV using FFmpeg)

Supported Speakers:
- NF: Northern Female (Miền Bắc - Nữ)
- SF: Southern Female (Miền Nam - Nữ)
- NM1: Northern Male 1 (Miền Bắc - Nam)
- NM2: Northern Male 2 (Miền Bắc - Nam)
- SM: Southern Male (Miền Nam - Nam)

Usage (curl examples):

# Create TTS job:
# curl -X POST http://localhost:8000/tts \\
#   -H "Content-Type: application/json" \\
#   -d '{
#     "text": "Xin chào, đây là test TTS local"
#   }'

# With optional parameters:
# curl -X POST http://localhost:8000/tts \\
#   -H "Content-Type: application/json" \\
#   -d '{
#     "text": "Xin chào các bạn",
#     "voice": "NF",
#     "speed": 1.0
#   }'

# Poll result:
# curl http://localhost:8000/job/<job_id>
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from job_manager import Job, JobManager

# Set up logging
logger = logging.getLogger(__name__)

# Global model instance - loaded once at startup
_tts_model = None
_model_loading_error = None

# Default parameters
DEFAULT_SPEAKER = "NF"  # Northern Female
DEFAULT_SPEED = 1.0
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_SCALE_W = 0.8

# Add valtec-tts to path
VALTEC_TTS_PATH = Path(__file__).parent / "valtec-tts"


def load_model(device: str = "cpu"):
    """
    Load the Valtec TTS model into memory.
    
    Called once at application startup.
    Model is reused across all requests to avoid repeated loading overhead.
    
    Args:
        device: Device to use ('cuda' or 'cpu'). Default is 'cpu' for VPS.
    
    Raises:
        RuntimeError: If model fails to load
    """
    global _tts_model, _model_loading_error
    
    if _tts_model is not None:
        logger.info("Valtec TTS model already loaded")
        return _tts_model
    
    print("Loading Valtec TTS model...")
    
    try:
        # Add valtec-tts to Python path
        if str(VALTEC_TTS_PATH) not in sys.path:
            sys.path.insert(0, str(VALTEC_TTS_PATH))
        
        # Import and initialize TTS
        from valtec_tts import TTS
        
        # Initialize TTS with specified device
        # Model will auto-download from Hugging Face if not cached
        _tts_model = TTS(device=device)
        
        print(f"Valtec TTS model loaded successfully!")
        print(f"  Device: {_tts_model.device}")
        print(f"  Available speakers: {_tts_model.speakers}")
        
        _model_loading_error = None
        return _tts_model
        
    except Exception as e:
        error_msg = f"Failed to load Valtec TTS model: {e}"
        print(error_msg)
        _model_loading_error = str(e)
        raise RuntimeError(error_msg)


def get_model():
    """Get the loaded TTS model instance."""
    if _tts_model is None:
        raise RuntimeError("Valtec TTS model not loaded. Call load_model() first.")
    return _tts_model


def get_available_speakers() -> list:
    """Get list of available speaker voices."""
    if _tts_model is not None:
        return _tts_model.speakers
    return ["NF", "SF", "NM1", "NM2", "SM"]


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    """
    Convert WAV file to MP3 using FFmpeg.
    
    Args:
        wav_path: Path to input WAV file
        mp3_path: Path to output MP3 file
        
    Raises:
        RuntimeError: If FFmpeg conversion fails
    """
    logger.info(f"Converting WAV to MP3: {wav_path} -> {mp3_path}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",  # High quality (VBR ~190kbps)
        str(mp3_path)
    ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.decode() if result.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg WAV to MP3 conversion failed: {error_msg}")
    
    logger.info(f"MP3 saved to: {mp3_path}")


def _synthesize_tts_sync(
    text: str,
    speaker: str,
    speed: float,
    output_wav_path: Path
) -> Tuple[np.ndarray, int]:
    """
    Synthesize speech from text (SYNCHRONOUS, blocking).
    
    Args:
        text: Vietnamese text to synthesize
        speaker: Speaker voice name (NF, SF, NM1, NM2, SM)
        speed: Speech speed (1.0 = normal, < 1.0 = faster, > 1.0 = slower)
        output_wav_path: Path to save WAV output
        
    Returns:
        Tuple of (audio_array, sample_rate)
        
    Raises:
        RuntimeError: If synthesis fails
    """
    logger.info(f"Synthesizing TTS: speaker={speaker}, speed={speed}, text={text[:50]}...")
    
    model = get_model()
    
    # Validate speaker
    if speaker not in model.speakers:
        logger.warning(f"Speaker '{speaker}' not found, using default: {model.speakers[0]}")
        speaker = model.speakers[0]
    
    # Synthesize audio
    audio, sr = model.synthesize(
        text=text,
        speaker=speaker,
        speed=speed,
        noise_scale=DEFAULT_NOISE_SCALE,
        noise_scale_w=DEFAULT_NOISE_SCALE_W
    )
    
    # Save WAV file
    import soundfile as sf
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav_path), audio, sr)
    
    logger.info(f"WAV saved to: {output_wav_path}")
    
    return audio, sr


def _process_local_tts_sync(job: Job, manager: JobManager) -> dict:
    """
    Process a local TTS job (SYNCHRONOUS, blocking).
    
    Args:
        job: Job instance with params:
            - text: Text to synthesize
            - voice: Optional speaker voice (default: NF)
            - speed: Optional speech speed (default: 1.0)
        manager: JobManager instance
        
    Returns:
        dict with 'file_url' pointing to the generated MP3 file
        
    Raises:
        RuntimeError: If synthesis or conversion fails
    """
    logger.info(f"Processing local TTS job {job.job_id}")
    
    # Check model is loaded
    if _tts_model is None:
        raise RuntimeError(
            "Valtec TTS model is not loaded. This should not happen - "
            "models are loaded during startup before accepting requests."
        )
    
    params = job.params
    text = params.get("text", "")
    
    # Validate text
    if not text or not text.strip():
        raise ValueError("Text is empty. Please provide text to synthesize.")
    
    # Get optional parameters with defaults
    speaker = params.get("voice", DEFAULT_SPEAKER)
    speed = params.get("speed", DEFAULT_SPEED)
    
    # Clamp speed to reasonable range
    speed = max(0.5, min(2.0, float(speed)))
    
    # File paths
    wav_filename = f"{job.job_id}.wav"
    mp3_filename = f"{job.job_id}.mp3"
    wav_path = job.work_dir / wav_filename
    mp3_path = manager.output_dir / mp3_filename
    
    # Step 1: Synthesize to WAV
    logger.info(f"Step 1: Synthesizing speech...")
    _synthesize_tts_sync(
        text=text.strip(),
        speaker=speaker,
        speed=speed,
        output_wav_path=wav_path
    )
    
    # Step 2: Convert WAV to MP3
    logger.info(f"Step 2: Converting to MP3...")
    _convert_wav_to_mp3(wav_path, mp3_path)
    
    # Verify output exists
    if not mp3_path.exists():
        raise RuntimeError(f"Failed to generate TTS output at {mp3_path}")
    
    logger.info(f"Local TTS job {job.job_id} completed successfully")
    
    # Cleanup working directory (WAV file)
    manager.cleanup_job_work_dir(job)
    
    return {
        "file_url": f"/static/{mp3_filename}"
    }


async def process_local_tts(job: Job, manager: JobManager) -> dict:
    """
    Process a local text-to-speech job using Valtec TTS (ASYNC wrapper).
    
    This function schedules blocking inference work in an executor
    to keep the event loop responsive.
    
    Steps:
    1. Validate input text (10% progress)
    2. Run TTS inference (30% progress)
    3. Convert WAV to MP3 (90% progress)
    4. Return result (100% progress)
    
    Args:
        job: Job instance with params:
            - text: Vietnamese text to synthesize
            - voice: Optional speaker voice (NF, SF, NM1, NM2, SM)
            - speed: Optional speech speed (1.0 = normal)
        manager: JobManager for progress updates
        
    Returns:
        dict with 'file_url' pointing to the generated MP3 file
        
    Example curl:
        # Create TTS job:
        curl -X POST http://localhost:8000/tts \\
          -H "Content-Type: application/json" \\
          -d '{
            "text": "Xin chào, đây là test TTS local"
          }'
        
        # Check job:
        curl http://localhost:8000/job/<job_id>
    """
    logger.info(f"Starting local TTS job {job.job_id}")
    
    # Update progress - preparing
    await manager.update_progress(job.job_id, 10)
    
    # Validate text early
    text = job.params.get("text", "")
    if not text or not text.strip():
        raise ValueError("Text is empty. Please provide text to synthesize.")
    
    # Update progress - starting inference
    await manager.update_progress(job.job_id, 30)
    
    # Schedule blocking TTS work in executor
    loop = asyncio.get_running_loop()
    
    logger.info("Scheduling Valtec TTS inference in executor")
    
    # Run blocking work in executor (default ThreadPoolExecutor)
    result = await loop.run_in_executor(
        None,
        _process_local_tts_sync,
        job,
        manager
    )
    
    # Update progress - saving file
    await manager.update_progress(job.job_id, 90)
    
    # Update progress - done
    await manager.update_progress(job.job_id, 100)
    
    logger.info(f"Local TTS job {job.job_id} completed")
    
    return result
