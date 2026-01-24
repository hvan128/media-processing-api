"""
Vocal Separation Service Module
===============================
Handles vocal separation using Demucs.

Design Decisions:
- Uses htdemucs_ft model with 2-stem separation (vocals/no_vocals)
- Model loaded once at startup to avoid repeated loading
- CPU-only operation with optimized parameters
- Only returns no_vocals track as per requirements
"""

import shutil
import subprocess
import asyncio
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from job_manager import Job, JobManager
from downloader import download_file


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


async def run_demucs_cli(audio_path: Path, output_dir: Path) -> Path:
    """
    Run Demucs separation using the CLI (fallback method).
    
    This uses subprocess to run Demucs, which is more reliable
    for CPU-only environments and handles memory better.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to write output
    
    Returns:
        Path to the no_vocals.wav file
    """
    # Build demucs command
    # -n htdemucs: Use htdemucs model
    # --two-stems vocals: Only separate vocals (creates vocals and no_vocals)
    # -d cpu: Force CPU device
    # --mp3: Output as MP3 (smaller files)
    # -o: Output directory
    
    cmd = [
        "python", "-m", "demucs",
        "-n", MODEL_NAME,
        "--two-stems", "vocals",
        "-d", DEVICE,
        "-o", str(output_dir),
        str(audio_path)
    ]
    
    # Run in subprocess
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise RuntimeError(f"Demucs failed: {error_msg}")
    
    # Demucs outputs to: output_dir/htdemucs/audio_filename/no_vocals.wav
    audio_stem = audio_path.stem
    no_vocals_path = output_dir / MODEL_NAME / audio_stem / "no_vocals.wav"
    
    if not no_vocals_path.exists():
        # Try .mp3 extension
        no_vocals_path = output_dir / MODEL_NAME / audio_stem / "no_vocals.mp3"
    
    if not no_vocals_path.exists():
        raise RuntimeError(f"Expected output file not found: {no_vocals_path}")
    
    return no_vocals_path


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
    params = job.params
    media_url = params["media_url"]
    
    # Step 1: Download audio file
    await manager.update_progress(job.job_id, 5)
    
    audio_path = await download_file(
        url=media_url,
        dest_dir=job.work_dir,
        filename="audio"
    )
    
    await manager.update_progress(job.job_id, 20)
    
    # Step 2: Run Demucs separation
    # Create a subdirectory for Demucs output
    demucs_output_dir = job.work_dir / "demucs_output"
    demucs_output_dir.mkdir(exist_ok=True)
    
    # Demucs doesn't provide progress callbacks easily,
    # so we'll update progress in steps
    await manager.update_progress(job.job_id, 30)
    
    # Run separation
    no_vocals_path = await run_demucs_cli(audio_path, demucs_output_dir)
    
    await manager.update_progress(job.job_id, 90)
    
    # Step 3: Copy to output directory
    output_filename = f"{job.job_id}_no_vocals.wav"
    output_path = manager.output_dir / output_filename
    
    shutil.copy2(no_vocals_path, output_path)
    
    await manager.update_progress(job.job_id, 100)
    
    # Cleanup working directory
    manager.cleanup_job_work_dir(job)
    
    return {
        "file_url": f"/static/{output_filename}"
    }
