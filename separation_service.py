"""
Vocal Separation Service Module
===============================
Handles speech suppression using FFmpeg's RNNoise-based filters, optimized for extracting industrial sounds.

Design Decisions:
- Uses FFmpeg's `arnndn` filter (RNNoise-based) as PRIMARY method for speech suppression
- Neural network-based approach provides superior speech removal vs frequency filtering
- Lightweight CPU-only operation, much faster than Spleeter
- Returns audio with speech removed (industrial sounds preserved)

Audio Processing Pipeline:
1. Preprocess audio to 48kHz mono (RNNoise requirement)
2. Apply FFmpeg's arnndn filter FIRST (neural speech suppression)
3. Apply light post-processing (mild high-pass to remove rumble only)
4. Preserve industrial sounds by using conservative filtering after arnndn

FFmpeg RNNoise Benefits:
- Neural network trained specifically for speech/noise separation
- ~10x faster than Spleeter
- No Python ML dependencies (~1.5GB savings)
- Real-time capable

Expected Performance:
- ~2-3 seconds for 1 minute audio on CPU
- Real-time capable for streaming applications

Limitations:
- Requires FFmpeg compiled with librnnoise support
- arnndn operates at 48kHz mono
- Model file must be available
"""

import shutil
import subprocess
import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import httpx

from job_manager import Job, JobManager
from downloader import download_file

# Set up logging
logger = logging.getLogger(__name__)

# RNNoise configuration (via FFmpeg)
# FFmpeg's arnndn filter operates at 48kHz mono
TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1

# Speech suppression parameters
# arnndn filter - PRIMARY method for speech suppression
ENABLE_ARNDN = True  # Enabled by default - use neural network suppression
ARNDN_MIX = 0.0  # Full suppression (0.0 = maximum noise/speech reduction)
RNNOISE_MODEL_URL = "https://github.com/xiph/rnnoise/raw/master/src/rnnoise_model.rnnn"

# Model file path
# Use MODEL_PATH env var if set (Docker), otherwise use app directory
_model_base = Path(os.environ.get("MODEL_PATH", "/app"))
RNNOISE_MODEL_PATH = _model_base / "rnnoise_model.rnnn"

# Post-processing filters (light, only for cleanup)
# Reduced from aggressive 5kHz to preserve industrial sounds
POST_HIGHPASS_FREQ = 2500  # Light high-pass: 2.5kHz (removes residual speech, preserves metal sounds)
RESIDUAL_HIGHPASS_FREQ = 100  # High-pass filter to remove low rumble only
ENABLE_RESIDUAL_FILTER = True  # Apply light post-processing

# Global state
_model_loaded = False
_model_loading_error = None
_arnndn_available = False


def _download_rnnoise_model():
    """
    Download RNNoise model file if it doesn't exist.
    
    Downloads the standard RNNoise model from xiph/rnnoise repository.
    This model is required for FFmpeg's arnndn filter.
    """
    if RNNOISE_MODEL_PATH.exists():
        logger.info(f"RNNoise model already exists at {RNNOISE_MODEL_PATH}")
        return
    
    logger.info(f"Downloading RNNoise model from: {RNNOISE_MODEL_URL}")
    
    # Create model directory if needed
    RNNOISE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download with redirect support
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            response = client.get(RNNOISE_MODEL_URL)
            response.raise_for_status()
            
            # Save model file
            with open(RNNOISE_MODEL_PATH, "wb") as f:
                f.write(response.content)
        
        logger.info(f"RNNoise model downloaded to: {RNNOISE_MODEL_PATH}")
        
        # Verify file exists and has content
        if not RNNOISE_MODEL_PATH.exists() or RNNOISE_MODEL_PATH.stat().st_size == 0:
            raise RuntimeError(f"Model download failed: file is empty or missing")
        
    except Exception as e:
        error_msg = f"Failed to download RNNoise model: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def load_model():
    """
    Initialize speech suppression system with RNNoise.
    
    Downloads RNNoise model file and verifies FFmpeg has arnndn filter support.
    """
    global _model_loaded, _model_loading_error, _arnndn_available
    
    if _model_loaded:
        return
    
    print("Initializing RNNoise-based speech suppression...")
    
    try:
        # Check if FFmpeg is available
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if result.returncode != 0:
            raise RuntimeError("FFmpeg is not available")
        
        # Check if arnndn filter is available
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        filters_output = result.stdout.decode() + result.stderr.decode()
        if "arnndn" in filters_output:
            _arnndn_available = True
            logger.info("FFmpeg arnndn filter is available")
            
            # Download RNNoise model file
            _download_rnnoise_model()
        else:
            _arnndn_available = False
            logger.warning(
                "FFmpeg arnndn filter not available. Speech suppression will use fallback method. "
                "For best results, compile FFmpeg with --enable-librnnoise"
            )
        
        _model_loaded = True
        _model_loading_error = None
        
        if _arnndn_available:
            print(f"RNNoise speech suppression initialized successfully (model: {RNNOISE_MODEL_PATH})")
        else:
            print("Speech suppression initialized (fallback mode - arnndn not available)")
        
    except FileNotFoundError:
        error_msg = "FFmpeg not found. Please install FFmpeg."
        print(error_msg)
        _model_loading_error = error_msg
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Error initializing speech suppression: {e}"
        print(error_msg)
        _model_loading_error = str(e)
        raise


def _preprocess_audio_sync(input_path: Path, output_path: Path) -> None:
    """
    Preprocess audio to 48kHz mono WAV for RNNoise processing.
    
    FFmpeg's arnndn filter requires 48kHz mono audio.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output preprocessed WAV
    """
    logger.info(f"Preprocessing audio for speech suppression: {input_path} -> {output_path}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ar", str(TARGET_SAMPLE_RATE),  # 48kHz (RNNoise requirement)
        "-ac", str(TARGET_CHANNELS),  # Mono (RNNoise requirement)
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
    
    logger.info(f"Preprocessed audio saved to: {output_path} (mono, {TARGET_SAMPLE_RATE}Hz)")


def _apply_speech_suppression_sync(input_path: Path, output_path: Path) -> None:
    """
    Apply speech suppression using FFmpeg's arnndn filter (RNNoise neural network).
    
    PRIMARY METHOD: arnndn filter with full suppression (m=0.0)
    This uses a neural network trained specifically for speech/noise separation.
    
    POST-PROCESSING: Light filtering only to remove residual artifacts
    - Mild high-pass at 2.5kHz (removes residual speech, preserves industrial sounds)
    - High-pass at 100Hz (removes rumble only)
    
    Args:
        input_path: Path to preprocessed audio (48kHz mono)
        output_path: Path to output with speech suppressed
    """
    logger.info(f"Applying RNNoise-based speech suppression: {input_path} -> {output_path}")
    
    # Build filter chain - arnndn FIRST, then light post-processing
    filters = []
    
    # Stage 1: PRIMARY - RNNoise neural network suppression
    if ENABLE_ARNDN and _arnndn_available:
        # Use external model file with full suppression (m=0.0)
        model_path_str = str(RNNOISE_MODEL_PATH.absolute())
        filters.append(f"arnndn=model={model_path_str}:m={ARNDN_MIX}")
        logger.info(f"Applying arnndn filter with model: {model_path_str}, mix={ARNDN_MIX} (full suppression)")
    else:
        logger.warning("arnndn filter not available, using fallback method")
        # Fallback: use high-pass filtering
        filters.append(f"highpass=f={POST_HIGHPASS_FREQ}")
        logger.info(f"Using fallback high-pass filter at {POST_HIGHPASS_FREQ}Hz")
    
    # Stage 2: POST-PROCESSING - Light filtering only
    # Mild high-pass to remove residual speech (preserves industrial sounds > 2.5kHz)
    if POST_HIGHPASS_FREQ > 0:
        filters.append(f"highpass=f={POST_HIGHPASS_FREQ}")
        logger.info(f"Applying post-processing high-pass at {POST_HIGHPASS_FREQ}Hz")
    
    # Stage 3: Remove rumble only
    filters.append(f"highpass=f={RESIDUAL_HIGHPASS_FREQ}")
    
    # Combine all filters
    filter_chain = ",".join(filters) if filters else "anull"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", filter_chain,
        "-acodec", "pcm_s16le",
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
        logger.warning(
            f"Speech suppression filter failed: {error_msg}, trying simplified approach"
        )
        # Fallback: use high-pass filter only
        _apply_highpass_only_sync(input_path, output_path)
    else:
        logger.info(f"Speech-suppressed audio saved to: {output_path}")


def _apply_highpass_only_sync(input_path: Path, output_path: Path) -> None:
    """
    Fallback: Apply high-pass filter only (when arnndn is not available).
    
    This is a simpler approach that removes low frequencies where speech
    fundamentals typically reside (300-3000Hz).
    
    Args:
        input_path: Path to input audio
        output_path: Path to output
    """
    logger.info(f"Applying high-pass filter (fallback): {input_path} -> {output_path}")
    
    # Use conservative high-pass to preserve industrial sounds
    highpass_freq = POST_HIGHPASS_FREQ
    
    filter_chain = f"highpass=f={highpass_freq},highpass=f={RESIDUAL_HIGHPASS_FREQ}"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", filter_chain,
        "-acodec", "pcm_s16le",
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
        raise RuntimeError(f"High-pass filter failed: {error_msg}")
    
    logger.info(f"High-pass filtered audio saved to: {output_path}")


def _apply_residual_filter_sync(input_path: Path, output_path: Path) -> None:
    """
    Apply residual filtering to remove any remaining speech artifacts.
    
    Uses FFmpeg filters:
    - Compressor to even out levels
    - High-pass at 100Hz to remove rumble
    """
    if not ENABLE_RESIDUAL_FILTER:
        shutil.copy2(input_path, output_path)
        return
    
    logger.info(f"Applying residual filtering: {input_path} -> {output_path}")
    
    # Filter chain:
    # 1. High-pass at 100Hz to remove rumble
    # 2. Compressor to even out levels
    filter_chain = (
        f"highpass=f={RESIDUAL_HIGHPASS_FREQ},"
        f"acompressor=threshold=-20dB:ratio=4:attack=5:release=50"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", filter_chain,
        "-acodec", "pcm_s16le",
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
        logger.warning(f"Residual filtering failed: {error_msg}, using unfiltered output")
        shutil.copy2(input_path, output_path)
    else:
        logger.info(f"Filtered audio saved to: {output_path}")


def _process_separation_sync(job: Job, manager: JobManager, audio_path: Path) -> dict:
    """
    Process a speech suppression job using FFmpeg's RNNoise filters (SYNCHRONOUS, blocking).
    
    Pipeline:
    1. Preprocess audio to 48kHz mono (RNNoise requirement)
    2. Apply FFmpeg's arnndn filter FIRST (neural speech suppression)
    3. Apply light post-processing (optional)
    4. Copy output to static directory
    
    Args:
        job: Job instance with params
        manager: JobManager instance (for cleanup)
        audio_path: Path to downloaded audio file
    
    Returns:
        dict with 'file_url' pointing to the processed audio file
    """
    logger.info(f"Processing separation job {job.job_id} with RNNoise")
    
    # Check model is ready
    if not _model_loaded:
        raise RuntimeError(
            "Speech suppression is not initialized. This should not happen - "
            "models are loaded during startup before accepting requests."
        )
    
    # Create working directories
    work_dir = job.work_dir
    preprocessed_path = work_dir / "preprocessed.wav"
    suppressed_path = work_dir / "speech_suppressed.wav"
    filtered_path = work_dir / "filtered.wav"
    
    # Step 1: Preprocess audio to 48kHz mono
    logger.info("Step 1: Preprocessing audio to 48kHz mono")
    _preprocess_audio_sync(audio_path, preprocessed_path)
    
    # Step 2: Apply RNNoise neural network suppression FIRST
    logger.info("Step 2: Applying RNNoise neural network speech suppression")
    _apply_speech_suppression_sync(preprocessed_path, suppressed_path)
    
    # Step 3: Apply residual filtering (optional, light post-processing)
    if ENABLE_RESIDUAL_FILTER:
        logger.info("Step 3: Applying light residual filtering")
        _apply_residual_filter_sync(suppressed_path, filtered_path)
        final_path = filtered_path
    else:
        final_path = suppressed_path
    
    # Step 4: Copy output to static directory
    output_filename = f"{job.job_id}_no_vocals.wav"
    output_path = manager.output_dir / output_filename
    
    logger.info(f"Copying output to: {output_path}")
    shutil.copy2(final_path, output_path)
    
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
    Process a speech suppression job using FFmpeg's RNNoise filters (ASYNC wrapper).
    
    This function schedules blocking work in an executor to keep the event loop responsive.
    Only I/O operations (download) remain async. All CPU-bound work runs in executor.
    
    Steps:
    1. Download audio file (async I/O, 0-20% progress)
    2. Schedule FFmpeg processing in executor (20-90% progress)
    3. Copy output to static directory (in executor, 90-100% progress)
    
    Args:
        job: Job instance with params:
            - media_url: URL to audio file
        manager: JobManager for progress updates
    
    Returns:
        dict with 'file_url' pointing to the processed audio file
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
    
    logger.info("Scheduling RNNoise speech suppression in executor")
    
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
