"""
Vocal Separation Service Module
===============================
Handles vocal separation using Spleeter.

Design Decisions:
- Uses Spleeter 2stems model (vocals/accompaniment)
- CPU-only operation, optimized for speed
- Only returns accompaniment track (no_vocals) as per requirements
- Uses Spleeter as PYTHON LIBRARY (not CLI) to avoid typer import issues
- Preprocessing: mono 16kHz for speed
- Expected: ~10 seconds for 1 minute audio on CPU
"""

import shutil
import subprocess
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from job_manager import Job, JobManager
from downloader import download_file

# Set up logging
logger = logging.getLogger(__name__)

# Spleeter configuration
MODEL_NAME = "spleeter:2stems"
# Target sample rate for CPU optimization
TARGET_SAMPLE_RATE = 16000

# Global separator instance
_separator = None
_model_loaded = False
_model_loading_error = None


def load_model():
    """
    Pre-load Spleeter model.
    
    Loads Spleeter separator at startup to avoid delays during job processing.
    Uses Python library directly (not CLI) to avoid typer import issues.
    """
    global _separator, _model_loaded, _model_loading_error
    
    if _model_loaded:
        return
    
    print(f"Initializing Spleeter ({MODEL_NAME}) for CPU...")
    
    try:
        # Set TensorFlow to CPU-only mode before importing
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
        
        # Import spleeter library directly (bypasses CLI and typer)
        from spleeter.separator import Separator
        
        # Create separator instance - this downloads the model
        _separator = Separator(MODEL_NAME)
        
        _model_loaded = True
        _model_loading_error = None
        print("Spleeter model loaded successfully")
        
    except Exception as e:
        error_msg = f"Error loading Spleeter model: {e}"
        print(error_msg)
        _model_loading_error = str(e)
        raise


def _preprocess_audio_sync(input_path: Path, output_path: Path) -> None:
    """
    Preprocess audio to mono 16kHz WAV using FFmpeg (SYNCHRONOUS).
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output preprocessed WAV
    """
    logger.info(f"Preprocessing audio: {input_path} -> {output_path}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",  # Mono
        "-ar", str(TARGET_SAMPLE_RATE),  # 16kHz
        "-acodec", "pcm_s16le",  # 16-bit PCM
        str(output_path)
    ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.decode() if result.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg preprocessing failed: {error_msg}")
    
    logger.info(f"Preprocessed audio saved to: {output_path}")


def _run_spleeter_sync(audio_path: Path, output_dir: Path) -> Path:
    """
    Run Spleeter separation using Python library (SYNCHRONOUS).
    
    Uses Spleeter as a library directly, bypassing CLI to avoid typer issues.
    
    Args:
        audio_path: Path to input audio file (preprocessed)
        output_dir: Directory to write output
    
    Returns:
        Path to the accompaniment.wav file (no_vocals)
    """
    global _separator
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running Spleeter separation on: {audio_path}")
    
    if _separator is None:
        raise RuntimeError("Spleeter separator not initialized")
    
    # Run separation using library API
    # separate_to_file writes output to output_dir/audio_stem/
    _separator.separate_to_file(
        str(audio_path),
        str(output_dir)
    )
    
    # Spleeter outputs to: output_dir/audio_stem/accompaniment.wav
    audio_stem = audio_path.stem
    accompaniment_path = output_dir / audio_stem / "accompaniment.wav"
    
    if not accompaniment_path.exists():
        # List files in output directory for debugging
        stem_dir = output_dir / audio_stem
        output_files = list(stem_dir.iterdir()) if stem_dir.exists() else []
        raise RuntimeError(
            f"Spleeter output not found: {accompaniment_path}. "
            f"Available files: {[f.name for f in output_files]}"
        )
    
    logger.info(f"Spleeter separation completed: {accompaniment_path}")
    
    return accompaniment_path


def _process_separation_sync(job: Job, manager: JobManager, audio_path: Path) -> dict:
    """
    Process a vocal separation job (SYNCHRONOUS, blocking).
    
    This function contains ALL CPU-bound and blocking operations:
    - Audio preprocessing with FFmpeg
    - Spleeter separation
    - File I/O operations
    
    This MUST run in an executor to avoid blocking the event loop.
    
    Args:
        job: Job instance with params
        manager: JobManager instance (for cleanup)
        audio_path: Path to downloaded audio file
    
    Returns:
        dict with 'file_url' pointing to the no_vocals audio file
    """
    logger.info(f"Processing separation job {job.job_id} (sync)")
    
    # Check model is ready
    if not _model_loaded:
        raise RuntimeError(
            "Spleeter model is not loaded. This should not happen - "
            "models are loaded during startup before accepting requests."
        )
    
    # Create working directories
    work_dir = job.work_dir
    preprocessed_path = work_dir / "preprocessed.wav"
    spleeter_output_dir = work_dir / "spleeter_output"
    
    # Step 1: Preprocess audio (mono 16kHz)
    logger.info("Preprocessing audio to mono 16kHz")
    _preprocess_audio_sync(audio_path, preprocessed_path)
    
    # Step 2: Run Spleeter separation
    logger.info("Running Spleeter separation")
    accompaniment_path = _run_spleeter_sync(preprocessed_path, spleeter_output_dir)
    
    # Step 3: Copy output to static directory
    output_filename = f"{job.job_id}_no_vocals.wav"
    output_path = manager.output_dir / output_filename
    
    logger.info(f"Copying output to: {output_path}")
    shutil.copy2(accompaniment_path, output_path)
    
    if not output_path.exists():
        raise RuntimeError(f"Failed to copy output file to {output_path}")
    
    logger.info(f"Separation job {job.job_id} completed successfully")
    
    # Cleanup working directory
    manager.cleanup_job_work_dir(job)
    
    return {
        "file_url": f"/static/{output_filename}"
    }


async def process_separation(job: Job, manager: JobManager) -> dict:
    """
    Process a vocal separation job (ASYNC wrapper).
    
    This function schedules blocking work in an executor to keep the event loop responsive.
    Only I/O operations (download) remain async. All CPU-bound work runs in executor.
    
    Steps:
    1. Download audio file (async I/O, 0-20% progress)
    2. Schedule Spleeter separation in executor (20-90% progress)
    3. Copy output to static directory (in executor, 90-100% progress)
    
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
    
    # Step 1: Download audio file (async I/O)
    await manager.update_progress(job.job_id, 5)
    logger.info(f"Downloading audio from: {media_url}")
    
    audio_path = await download_file(
        url=media_url,
        dest_dir=job.work_dir,
        filename="audio"
    )
    
    logger.info(f"Audio downloaded to: {audio_path}")
    await manager.update_progress(job.job_id, 20)
    
    # Step 2: Schedule ALL blocking work in executor
    loop = asyncio.get_running_loop()
    
    await manager.update_progress(job.job_id, 30)
    
    logger.info("Scheduling blocking separation work in executor")
    
    # Run blocking work in executor (default ThreadPoolExecutor)
    result = await loop.run_in_executor(
        None,
        _process_separation_sync,
        job,
        manager,
        audio_path
    )
    
    # Update progress after executor completes
    await manager.update_progress(job.job_id, 100)
    
    logger.info(f"Separation job {job.job_id} completed")
    
    return result
