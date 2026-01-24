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
# Using htdemucs with 2 stems for faster processing
MODEL_NAME = "htdemucs"
DEVICE = "cpu"

# Global model instance and apply function
_separator = None
_apply_model = None
_model_loaded = False


def load_model():
    """
    Pre-load Demucs model into memory.
    
    Loads Demucs model as a library (not CLI) to avoid torchaudio.save() calls.
    """
    global _model_loaded, _separator, _apply_model
    
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
        print("Demucs model loaded successfully (library mode)")
        
    except Exception as e:
        print(f"Error loading Demucs model: {e}")
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


async def load_audio_with_ffmpeg(audio_path: Path) -> Tuple[torch.Tensor, int]:
    """
    Load audio file using FFmpeg and convert to torch tensor.
    
    This bypasses torchaudio entirely and uses FFmpeg to decode audio.
    
    Args:
        audio_path: Path to input audio file (any format)
    
    Returns:
        Tuple of (audio_tensor, sample_rate)
        audio_tensor: Shape (channels, samples) as float32 tensor
        sample_rate: Sample rate in Hz
    """
    logger.info(f"Loading audio with FFmpeg: {audio_path}")
    
    # Use FFmpeg to decode audio to raw PCM (16-bit signed little-endian)
    # Output format: -f s16le = signed 16-bit little-endian PCM
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "s16le",  # Raw PCM format
        "-acodec", "pcm_s16le",
        "-ar", "44100",  # Resample to 44.1kHz (Demucs standard)
        "-ac", "2",  # Convert to stereo (2 channels)
        "-"  # Output to stdout
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg audio loading failed: {error_msg}")
    
    # Convert raw PCM bytes to numpy array
    # s16le = signed 16-bit little-endian, so we use int16
    audio_data = np.frombuffer(stdout, dtype=np.int16)
    
    # Reshape to (channels, samples)
    # We specified 2 channels, so reshape accordingly
    num_channels = 2
    num_samples = len(audio_data) // num_channels
    audio_data = audio_data[:num_samples * num_channels].reshape(num_channels, num_samples)
    
    # Convert to float32 and normalize to [-1.0, 1.0]
    audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
    
    sample_rate = 44100  # We resampled to 44.1kHz
    
    logger.info(f"Loaded audio: shape={audio_tensor.shape}, sample_rate={sample_rate}")
    
    return audio_tensor, sample_rate


async def save_audio_with_ffmpeg(audio_tensor: torch.Tensor, sample_rate: int, output_path: Path) -> None:
    """
    Save audio tensor to WAV file using FFmpeg.
    
    This bypasses torchaudio entirely and uses FFmpeg to encode audio.
    
    Args:
        audio_tensor: Audio tensor with shape (channels, samples) as float32
        sample_rate: Sample rate in Hz
        output_path: Path to output WAV file
    """
    logger.info(f"Saving audio with FFmpeg: {output_path}, shape={audio_tensor.shape}, sr={sample_rate}")
    
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
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate(input=audio_bytes)
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg audio saving failed: {error_msg}")
    
    if not output_path.exists():
        raise RuntimeError(f"FFmpeg saving completed but output file not found: {output_path}")
    
    logger.info(f"Successfully saved audio: {output_path}")


async def run_demucs_library(audio_path: Path, output_dir: Path) -> Path:
    """
    Run Demucs separation using the library API (NOT CLI).
    
    This avoids torchaudio.save() calls entirely by:
    1. Loading audio with FFmpeg
    2. Running Demucs inference in memory
    3. Extracting no_vocals source
    4. Saving with FFmpeg
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to write output
    
    Returns:
        Path to the no_vocals.wav file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running Demucs separation (library mode) on: {audio_path}")
    
    # Step 1: Load audio using FFmpeg (no torchaudio)
    audio_tensor, sample_rate = await load_audio_with_ffmpeg(audio_path)
    
    # Step 2: Get model and apply function
    model = get_model()
    apply_model = get_apply_model()
    
    # Step 3: Prepare audio for Demucs
    # Demucs expects shape (batch, channels, samples)
    # Add batch dimension
    audio_batch = audio_tensor.unsqueeze(0)  # Shape: (1, channels, samples)
    
    logger.info(f"Running Demucs inference on audio shape: {audio_batch.shape}")
    
    # Step 4: Run Demucs inference
    # Use full separation and combine non-vocal sources to get no_vocals
    # This avoids needing two_stems parameter which may not be available in apply_model
    with torch.no_grad():
        sources = apply_model(
            model,
            audio_batch,
            device=DEVICE,
            shifts=1,  # Number of random shifts for better quality
            split=True,  # Split long audio into chunks
            overlap=0.25,  # Overlap between chunks
            progress=False
        )
        
        # Sources shape: (batch, sources, channels, samples)
        # For htdemucs, sources order is typically: [drums, bass, other, vocals]
        # no_vocals = drums + bass + other (everything except vocals)
        if sources.shape[1] == 4:
            # Standard 4-source separation: drums, bass, other, vocals
            no_vocals = sources[0, 0:3].sum(dim=0)  # Sum drums, bass, other
        elif sources.shape[1] == 2:
            # Two-stem separation: assume index 1 is no_vocals
            no_vocals = sources[0, 1]
        else:
            # Fallback: sum all sources except the last (assuming last is vocals)
            logger.warning(f"Unexpected number of sources: {sources.shape[1]}, using fallback")
            no_vocals = sources[0, :-1].sum(dim=0)
    
    logger.info(f"Demucs inference completed. no_vocals shape: {no_vocals.shape}")
    
    # Step 5: Save no_vocals using FFmpeg (no torchaudio)
    output_path = output_dir / "no_vocals.wav"
    await save_audio_with_ffmpeg(no_vocals, sample_rate, output_path)
    
    if not output_path.exists():
        raise RuntimeError(f"Output file not found after saving: {output_path}")
    
    logger.info(f"Successfully created no_vocals.wav: {output_path}")
    
    return output_path


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
    
    # Step 2: Run Demucs separation (library mode, no CLI)
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
    
    # Run separation using library (NOT CLI - avoids torchaudio.save())
    logger.info("Starting Demucs separation process (library mode)")
    no_vocals_path = await run_demucs_library(audio_path, demucs_output_dir)
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
