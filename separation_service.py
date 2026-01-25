"""
Vocal Separation Service Module
===============================
Handles vocal separation using Demucs.

Design Decisions:
- Uses htdemucs model with 2-stem separation (vocals/no_vocals)
- Model loaded once at startup to avoid repeated loading
- CPU-only operation with optimized parameters
- Only returns no_vocals track as per requirements
- Uses Demucs as a LIBRARY (not CLI) to avoid torchaudio.save() calls
- Uses FFmpeg for audio loading and saving to avoid torchaudio backends
"""

import shutil
import subprocess
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import torch

from job_manager import Job, JobManager
from downloader import download_file

# Set up logging
logger = logging.getLogger(__name__)


# Demucs configuration
# Using classic "demucs" model - NOT htdemucs or htdemucs_ft (transformers are too slow on CPU)
# Full 4-stem separation, then compute no_vocals = drums + bass + other
# Optimized for CPU: mono input, 16kHz sample rate, no shifts, no overlap
MODEL_NAME = "demucs"
DEVICE = "cpu"
# Target sample rate for CPU optimization (16kHz for speed)
TARGET_SAMPLE_RATE = 16000

# Global model instance and apply function
_separator = None
_apply_model = None
_model_loaded = False
_model_loading_error = None


def load_model():
    """
    Pre-load Demucs model into memory.
    
    Loads Demucs model as a library (not CLI) to avoid torchaudio.save() calls.
    """
    global _model_loaded, _separator, _apply_model, _model_loading_error
    
    if _model_loaded:
        return
    
    print(f"Initializing Demucs ({MODEL_NAME}) as library for CPU...")
    
    # Set torch to use CPU
    torch.set_num_threads(4)  # Reasonable thread count for VPS
    
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        
        # Load model to trigger download
        model = get_model(MODEL_NAME)
        model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        
        # Store for reuse
        _separator = model
        _apply_model = apply_model
        
        _model_loaded = True
        _model_loading_error = None  # Clear any previous error
        print("Demucs model loaded successfully (library mode)")
        
    except Exception as e:
        error_msg = f"Error loading Demucs model: {e}"
        print(error_msg)
        _model_loading_error = str(e)
        raise


def get_model():
    """Get the loaded Demucs model"""
    if not _model_loaded:
        raise RuntimeError("Demucs model not loaded. Call load_model() first.")
    return _separator


def get_apply_model():
    """Get the apply_model function"""
    if not _model_loaded:
        raise RuntimeError("Demucs model not loaded. Call load_model() first.")
    return _apply_model


def _load_audio_with_ffmpeg_sync(audio_path: Path) -> Tuple[torch.Tensor, int]:
    """
    Load audio file using FFmpeg and convert to torch tensor (SYNCHRONOUS).
    
    OPTIMIZED for CPU speed:
    - Converts to MONO (1 channel) for faster processing
    - Resamples to 16kHz for faster inference
    
    This is a blocking operation that must run in an executor.
    Uses subprocess.run (not asyncio) to avoid blocking the event loop.
    
    Args:
        audio_path: Path to input audio file (any format)
    
    Returns:
        Tuple of (audio_tensor, sample_rate)
        audio_tensor: Shape (1, samples) as float32 tensor (mono)
        sample_rate: Sample rate in Hz (TARGET_SAMPLE_RATE)
    """
    logger.info(f"Loading audio with FFmpeg (sync, optimized for CPU): {audio_path}")
    
    # Use FFmpeg to decode and preprocess audio:
    # - Convert to MONO (1 channel) for speed
    # - Resample to TARGET_SAMPLE_RATE (16kHz) for speed
    # - Output as raw PCM (16-bit signed little-endian)
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "s16le",  # Raw PCM format
        "-acodec", "pcm_s16le",
        "-ar", str(TARGET_SAMPLE_RATE),  # Resample to target rate (16kHz)
        "-ac", "1",  # Convert to MONO (1 channel) for speed
        "-"  # Output to stdout
    ]
    
    # Use subprocess.run (blocking, but runs in executor thread)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.decode() if result.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg audio loading failed: {error_msg}")
    
    # Convert raw PCM bytes to numpy array
    # s16le = signed 16-bit little-endian, so we use int16
    audio_data = np.frombuffer(result.stdout, dtype=np.int16)
    
    # Reshape to (1, samples) for mono
    audio_data = audio_data.reshape(1, -1)
    
    # Convert to float32 and normalize to [-1.0, 1.0]
    audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
    
    logger.info(f"Loaded audio (mono, {TARGET_SAMPLE_RATE}Hz): shape={audio_tensor.shape}, sample_rate={TARGET_SAMPLE_RATE}")
    
    return audio_tensor, TARGET_SAMPLE_RATE


def _save_audio_with_ffmpeg_sync(audio_tensor: torch.Tensor, sample_rate: int, output_path: Path) -> None:
    """
    Save audio tensor to WAV file using FFmpeg (SYNCHRONOUS).
    
    This is a blocking operation that must run in an executor.
    Uses subprocess.run (not asyncio) to avoid blocking the event loop.
    
    Args:
        audio_tensor: Audio tensor with shape (channels, samples) as float32
        sample_rate: Sample rate in Hz
        output_path: Path to output WAV file
    """
    logger.info(f"Saving audio with FFmpeg (sync): {output_path}, shape={audio_tensor.shape}, sr={sample_rate}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensor to numpy and denormalize to int16
    audio_np = audio_tensor.cpu().numpy()
    # Clamp to [-1, 1] range
    audio_np = np.clip(audio_np, -1.0, 1.0)
    # Convert to int16
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    
    # Convert to bytes (little-endian)
    audio_bytes = audio_int16.tobytes()
    
    # Use FFmpeg to encode to WAV
    # Input: raw PCM (s16le) from stdin
    # Output: WAV file
    cmd = [
        "ffmpeg", "-y",  # Overwrite output
        "-f", "s16le",  # Input format: signed 16-bit little-endian
        "-ar", str(sample_rate),  # Sample rate
        "-ac", str(audio_tensor.shape[0]),  # Number of channels
        "-i", "-",  # Read from stdin
        "-acodec", "pcm_s16le",  # Output codec
        str(output_path)
    ]
    
    # Use subprocess.run (blocking, but runs in executor thread)
    result = subprocess.run(
        cmd,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.decode() if result.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg audio saving failed: {error_msg}")
    
    if not output_path.exists():
        raise RuntimeError(f"FFmpeg saving completed but output file not found: {output_path}")
    
    logger.info(f"Successfully saved audio: {output_path}")


def _run_demucs_library_sync(audio_path: Path, output_dir: Path) -> Path:
    """
    Run Demucs separation using the library API (SYNCHRONOUS).
    
    This is a blocking operation that must run in an executor.
    Contains ALL CPU-bound work: FFmpeg, Demucs inference, torch operations.
    
    This avoids torchaudio.save() calls entirely by:
    1. Loading audio with FFmpeg (sync)
    2. Running Demucs inference in memory
    3. Extracting no_vocals source
    4. Saving with FFmpeg (sync)
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to write output
    
    Returns:
        Path to the no_vocals.wav file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running Demucs separation (sync, library mode) on: {audio_path}")
    
    # Step 1: Load audio using FFmpeg (synchronous, blocking)
    audio_tensor, sample_rate = _load_audio_with_ffmpeg_sync(audio_path)
    
    # Step 2: Get model and apply function
    model = get_model()
    apply_model = get_apply_model()
    
    # Step 3: Prepare audio for Demucs
    # Demucs expects shape (batch, channels, samples)
    # Add batch dimension
    audio_batch = audio_tensor.unsqueeze(0)  # Shape: (1, channels, samples)
    
    logger.info(f"Running Demucs inference on audio shape: {audio_batch.shape}")
    
    # Step 4: Run Demucs inference (CPU-bound, blocking)
    # OPTIMIZED for CPU speed:
    # - shifts=0: No random shifts (faster, deterministic)
    # - split=False: Process entire audio at once (faster for short audio)
    # - overlap=0.0: No overlap (faster)
    # - Full 4-stem separation, then compute no_vocals manually
    with torch.no_grad():
        sources = apply_model(
            model,
            audio_batch,
            device=DEVICE,
            shifts=0,  # No shifts for speed
            split=False,  # No splitting for speed
            overlap=0.0,  # No overlap for speed
            progress=False
        )
        
        # Sources shape: (batch, 4, channels, samples)
        # Demucs sources order: [drums, bass, other, vocals]
        # no_vocals = drums + bass + other (everything except vocals)
        logger.info(f"Sources shape: {sources.shape}")
        
        if sources.shape[1] != 4:
            raise RuntimeError(f"Expected 4 sources from demucs model, got {sources.shape[1]}")
        
        # Sum drums (0) + bass (1) + other (2) to get no_vocals
        no_vocals = sources[0, 0:3].sum(dim=0)
    
    logger.info(f"Demucs inference completed. no_vocals shape: {no_vocals.shape}")
    
    # Step 5: Save no_vocals using FFmpeg (synchronous, blocking)
    output_path = output_dir / "no_vocals.wav"
    _save_audio_with_ffmpeg_sync(no_vocals, sample_rate, output_path)
    
    if not output_path.exists():
        raise RuntimeError(f"Output file not found after saving: {output_path}")
    
    logger.info(f"Successfully created no_vocals.wav: {output_path}")
    
    return output_path


def _process_separation_sync(job: Job, manager: JobManager, audio_path: Path) -> dict:
    """
    Process a vocal separation job (SYNCHRONOUS, blocking).
    
    This function contains ALL CPU-bound and blocking operations:
    - Demucs inference (torch operations)
    - FFmpeg audio loading/saving
    - File I/O operations
    
    This MUST run in an executor to avoid blocking the event loop.
    Progress updates are handled by the async wrapper, not here.
    
    Args:
        job: Job instance with params
        manager: JobManager instance (for cleanup, not progress updates)
        audio_path: Path to downloaded audio file
    
    Returns:
        dict with 'file_url' pointing to the no_vocals audio file
    """
    logger.info(f"Processing separation job {job.job_id} (sync)")
    
    # Model is guaranteed to be loaded during FastAPI startup (blocking)
    # No waiting/polling needed here - if model isn't loaded, it's a fatal error
    if not _model_loaded:
        raise RuntimeError(
            "Demucs model is not loaded. This should not happen - "
            "models are loaded during startup before accepting requests."
        )
    
    # Create a subdirectory for Demucs output
    demucs_output_dir = job.work_dir / "demucs_output"
    demucs_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Demucs output directory: {demucs_output_dir}")
    
    # Run separation using library (synchronous, blocking)
    # This contains ALL CPU-bound work: FFmpeg, Demucs inference, torch
    logger.info("Starting Demucs separation process (sync, library mode)")
    no_vocals_path = _run_demucs_library_sync(audio_path, demucs_output_dir)
    logger.info(f"Demucs separation completed. Output: {no_vocals_path}")
    
    # Copy to output directory (blocking file I/O)
    output_filename = f"{job.job_id}_no_vocals.wav"
    output_path = manager.output_dir / output_filename
    
    logger.info(f"Copying output to: {output_path}")
    shutil.copy2(no_vocals_path, output_path)
    
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
    2. Schedule Demucs separation in executor (20-90% progress)
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
    
    # Step 1: Download audio file (async I/O - this is fine, doesn't block CPU)
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
    # This includes: Demucs inference, FFmpeg operations, file I/O
    # The event loop remains responsive while this runs in a thread
    loop = asyncio.get_running_loop()
    
    await manager.update_progress(job.job_id, 30)
    
    logger.info("Scheduling blocking separation work in executor")
    
    # Run blocking work in executor (default ThreadPoolExecutor)
    # This immediately yields control back to the event loop
    result = await loop.run_in_executor(
        None,  # Use default ThreadPoolExecutor
        _process_separation_sync,
        job,
        manager,
        audio_path
    )
    
    # Update progress after executor completes
    await manager.update_progress(job.job_id, 100)
    
    logger.info(f"Separation job {job.job_id} completed")
    
    return result
