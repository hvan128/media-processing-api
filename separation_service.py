"""
Vocal Separation Service Module
===============================
Handles vocal separation using Demucs.

Design Decisions:
- Uses htdemucs_ft model with 2-stem separation (vocals/no_vocals)
- Model loaded once at startup to avoid repeated loading
- CPU-only operation with optimized parameters
- Only returns no_vocals track as per requirements
- Uses FFmpeg for audio conversion to avoid torchaudio backend dependencies
"""

import shutil
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import Optional

import torch

from job_manager import Job, JobManager
from downloader import download_file

# Set up logging
logger = logging.getLogger(__name__)


# Demucs configuration
# Using htdemucs with 2 stems for faster processing
MODEL_NAME = "htdemucs"
DEVICE = "cpu"

# Global model instance
_separator = None
_model_loaded = False


def load_model():
    """
    Pre-load Demucs model into memory.
    
    Demucs uses a different loading pattern - we'll use the CLI approach
    which handles model caching automatically. This function ensures
    the model is downloaded and cached on first run.
    """
    global _model_loaded
    
    if _model_loaded:
        return
    
    print(f"Initializing Demucs ({MODEL_NAME}) for CPU...")
    
    # Set torch to use CPU
    torch.set_num_threads(4)  # Reasonable thread count for VPS
    
    # Demucs will download/cache the model on first use
    # We can pre-trigger this by importing the module
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        
        # Load model to trigger download
        model = get_model(MODEL_NAME)
        model.to(DEVICE)
        
        # Store for reuse
        global _separator
        _separator = model
        
        _model_loaded = True
        print("Demucs model loaded successfully")
        
    except Exception as e:
        print(f"Error loading Demucs model: {e}")
        raise


def get_model():
    """Get the loaded Demucs model"""
    if not _model_loaded:
        raise RuntimeError("Demucs model not loaded. Call load_model() first.")
    return _separator


async def run_ffmpeg_convert(input_path: Path, output_path: Path) -> None:
    """
    Convert audio file to WAV using FFmpeg.
    
    This bypasses torchaudio entirely and uses FFmpeg directly.
    
    Args:
        input_path: Path to input audio file (any format)
        output_path: Path to output WAV file
    """
    logger.info(f"Converting {input_path} to WAV using FFmpeg: {output_path}")
    
    cmd = [
        "ffmpeg", "-y",  # -y to overwrite output
        "-i", str(input_path),
        "-acodec", "pcm_s16le",  # 16-bit PCM WAV
        "-ar", "44100",  # 44.1kHz sample rate
        str(output_path)
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg conversion failed: {error_msg}")
    
    if not output_path.exists():
        raise RuntimeError(f"FFmpeg conversion completed but output file not found: {output_path}")
    
    logger.info(f"Successfully converted to WAV: {output_path}")


async def run_demucs_cli(audio_path: Path, output_dir: Path) -> Path:
    """
    Run Demucs separation using the CLI.
    
    This uses subprocess to run Demucs, which is more reliable
    for CPU-only environments and handles memory better.
    
    After Demucs finishes, converts output to WAV using FFmpeg
    to avoid torchaudio backend dependencies.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to write output
    
    Returns:
        Path to the no_vocals.wav file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build demucs command
    # -n htdemucs: Use htdemucs model
    # --two-stems vocals: Only separate vocals (creates vocals and no_vocals)
    # -d cpu: Force CPU device
    # -o: Output directory
    # Note: We let Demucs output in its default format, then convert with FFmpeg
    
    cmd = [
        "python", "-m", "demucs",
        "-n", MODEL_NAME,
        "--two-stems", "vocals",
        "-d", DEVICE,
        "-o", str(output_dir),
        str(audio_path)
    ]
    
    logger.info(f"Running Demucs command: {' '.join(cmd)}")
    
    # Run in subprocess
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    stdout_text = stdout.decode() if stdout else ""
    stderr_text = stderr.decode() if stderr else ""
    
    logger.info(f"Demucs stdout: {stdout_text[:500]}")  # Log first 500 chars
    if stderr_text:
        logger.info(f"Demucs stderr: {stderr_text[:500]}")
    
    if process.returncode != 0:
        error_msg = stderr_text if stderr_text else "Unknown error"
        raise RuntimeError(f"Demucs process failed (code {process.returncode}): {error_msg}")
    
    # Demucs outputs to: output_dir/htdemucs/audio_filename/no_vocals.*
    audio_stem = audio_path.stem
    demucs_output_subdir = output_dir / MODEL_NAME / audio_stem
    
    logger.info(f"Checking for Demucs output in: {demucs_output_subdir}")
    
    if not demucs_output_subdir.exists():
        raise RuntimeError(f"Demucs output directory not found: {demucs_output_subdir}")
    
    # List all files in the output directory
    output_files = list(demucs_output_subdir.iterdir())
    logger.info(f"Files found in Demucs output directory: {[f.name for f in output_files]}")
    
    # Look for no_vocals file (could be .wav, .mp3, or other format)
    no_vocals_candidates = [
        demucs_output_subdir / "no_vocals.wav",
        demucs_output_subdir / "no_vocals.mp3",
        demucs_output_subdir / "no_vocals.flac",
    ]
    
    # Also check for any file starting with "no_vocals"
    for file in output_files:
        if file.stem == "no_vocals" and file not in no_vocals_candidates:
            no_vocals_candidates.append(file)
    
    # Find the actual no_vocals file
    no_vocals_input = None
    for candidate in no_vocals_candidates:
        if candidate.exists():
            no_vocals_input = candidate
            logger.info(f"Found no_vocals file: {no_vocals_input}")
            break
    
    if no_vocals_input is None:
        raise RuntimeError(
            f"no_vocals file not found in {demucs_output_subdir}. "
            f"Available files: {[f.name for f in output_files]}"
        )
    
    # Always convert to WAV using FFmpeg to avoid torchaudio dependencies
    # This ensures we have a reliable WAV output regardless of Demucs output format
    no_vocals_wav = demucs_output_subdir / "no_vocals.wav"
    
    # If it's already WAV and we trust it, we could skip conversion
    # But to be safe and ensure compatibility, we always convert
    if no_vocals_input.suffix.lower() == ".wav":
        # Still convert to ensure proper WAV format and avoid torchaudio issues
        temp_wav = demucs_output_subdir / "no_vocals_temp.wav"
        await run_ffmpeg_convert(no_vocals_input, temp_wav)
        # Replace original with converted version
        if no_vocals_input != temp_wav:
            no_vocals_input.unlink()  # Remove original
            temp_wav.rename(no_vocals_wav)  # Rename temp to final
    else:
        # Convert from non-WAV format to WAV
        await run_ffmpeg_convert(no_vocals_input, no_vocals_wav)
        # Optionally remove original non-WAV file to save space
        if no_vocals_input != no_vocals_wav:
            no_vocals_input.unlink()
    
    if not no_vocals_wav.exists():
        raise RuntimeError(f"Final no_vocals.wav file not found after conversion: {no_vocals_wav}")
    
    logger.info(f"Successfully created no_vocals.wav: {no_vocals_wav}")
    
    return no_vocals_wav


async def process_separation(job: Job, manager: JobManager) -> dict:
    """
    Process a vocal separation job.
    
    Steps:
    1. Download audio file (0-20% progress)
    2. Run Demucs separation (20-90% progress)
    3. Copy output to static directory (90-100% progress)
    
    Args:
        job: Job instance with params:
            - media_url: URL to audio file
        manager: JobManager for progress updates
    
    Returns:
        dict with 'file_url' pointing to the no_vocals audio file
    """
    logger.info(f"Starting separation job {job.job_id}")
    
    params = job.params
    media_url = params["media_url"]
    
    # Step 1: Download audio file
    await manager.update_progress(job.job_id, 5)
    logger.info(f"Downloading audio from: {media_url}")
    
    audio_path = await download_file(
        url=media_url,
        dest_dir=job.work_dir,
        filename="audio"
    )
    
    logger.info(f"Audio downloaded to: {audio_path}")
    await manager.update_progress(job.job_id, 20)
    
    # Step 2: Run Demucs separation
    # Wait for model to be ready (models load in background during startup)
    # Models can take 30-90 seconds to load, so we wait up to 120 seconds
    import time
    max_wait = 120
    start_time = time.time()
    while not _model_loaded and (time.time() - start_time) < max_wait:
        await asyncio.sleep(1)
    
    if not _model_loaded:
        raise RuntimeError("Demucs model is not ready. Please try again in a moment.")
    
    # Create a subdirectory for Demucs output
    demucs_output_dir = job.work_dir / "demucs_output"
    demucs_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Demucs output directory: {demucs_output_dir}")
    
    # Demucs doesn't provide progress callbacks easily,
    # so we'll update progress in steps
    await manager.update_progress(job.job_id, 30)
    
    # Run separation
    logger.info("Starting Demucs separation process")
    no_vocals_path = await run_demucs_cli(audio_path, demucs_output_dir)
    logger.info(f"Demucs separation completed. Output: {no_vocals_path}")
    
    await manager.update_progress(job.job_id, 90)
    
    # Step 3: Copy to output directory
    output_filename = f"{job.job_id}_no_vocals.wav"
    output_path = manager.output_dir / output_filename
    
    logger.info(f"Copying output to: {output_path}")
    shutil.copy2(no_vocals_path, output_path)
    
    if not output_path.exists():
        raise RuntimeError(f"Failed to copy output file to {output_path}")
    
    logger.info(f"Separation job {job.job_id} completed successfully")
    await manager.update_progress(job.job_id, 100)
    
    # Cleanup working directory
    manager.cleanup_job_work_dir(job)
    
    return {
        "file_url": f"/static/{output_filename}"
    }
