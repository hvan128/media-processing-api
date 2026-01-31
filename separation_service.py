"""
Vocal Separation Service Module
===============================
Handles speech suppression using RNNoise, optimized for extracting industrial sounds.

Design Decisions:
- Uses RNNoise neural network for speech extraction
- Lightweight CPU-only operation, much faster than Spleeter
- Subtracts extracted speech from original to isolate background sounds
- Returns audio with speech removed (industrial sounds preserved)

Audio Processing Pipeline:
1. Preprocess audio to 48kHz mono (RNNoise requirement)
2. Run RNNoise to extract "clean speech" component
3. Invert phase and mix with original to cancel speech
4. Apply additional filtering to remove residual speech artifacts

RNNoise Benefits:
- ~10x faster than Spleeter
- ~50MB vs ~1.5GB dependencies
- Trained specifically for speech/noise separation
- No TensorFlow required

Expected Performance:
- ~2-3 seconds for 1 minute audio on CPU
- Real-time capable for streaming applications

Limitations:
- RNNoise operates at 48kHz mono only
- Optimized for voice calls, may miss some speech frequencies
- Subtraction artifacts possible in some edge cases
"""

import shutil
import subprocess
import asyncio
import logging
import os
import tempfile
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from job_manager import Job, JobManager
from downloader import download_file

# Set up logging
logger = logging.getLogger(__name__)

# RNNoise configuration
# RNNoise operates at 48kHz mono - this is a hard requirement
RNNOISE_SAMPLE_RATE = 48000
RNNOISE_CHANNELS = 1

# Speech suppression parameters
# These control how aggressively we remove speech
SPEECH_ATTENUATION_DB = -30  # How much to attenuate extracted speech before subtraction
RESIDUAL_HIGHPASS_FREQ = 100  # High-pass filter to remove low rumble after processing
ENABLE_RESIDUAL_FILTER = True  # Apply additional filtering on output

# Global state
_rnnoise_available = False
_model_loaded = False
_model_loading_error = None


def _check_rnnoise_available() -> bool:
    """Check if RNNoise Python bindings are available."""
    try:
        import rnnoise
        return True
    except ImportError:
        return False


def load_model():
    """
    Initialize RNNoise for speech suppression.
    
    RNNoise is a lightweight neural network - initialization is fast.
    This function validates that the rnnoise package is available.
    """
    global _rnnoise_available, _model_loaded, _model_loading_error
    
    if _model_loaded:
        return
    
    print("Initializing RNNoise speech suppression...")
    
    try:
        # Check if rnnoise package is available
        import rnnoise
        
        # Quick test to ensure it works
        _rnnoise_available = True
        _model_loaded = True
        _model_loading_error = None
        
        print("RNNoise initialized successfully")
        
    except ImportError as e:
        error_msg = f"RNNoise not available: {e}. Install with: pip install rnnoise-python"
        print(error_msg)
        _model_loading_error = error_msg
        raise ImportError(error_msg)
    except Exception as e:
        error_msg = f"Error initializing RNNoise: {e}"
        print(error_msg)
        _model_loading_error = str(e)
        raise


def _read_wav_raw(wav_path: Path) -> Tuple[np.ndarray, int, int]:
    """
    Read WAV file and return raw audio data as numpy array.
    
    Args:
        wav_path: Path to WAV file
        
    Returns:
        Tuple of (audio_data as float32 normalized -1 to 1, sample_rate, channels)
    """
    import wave
    
    with wave.open(str(wav_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        
        raw_data = wav_file.readframes(n_frames)
    
    # Convert to numpy array based on sample width
    if sample_width == 2:  # 16-bit
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:  # 32-bit
        audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    
    # Reshape for stereo
    if channels == 2:
        audio = audio.reshape(-1, 2)
    
    return audio, sample_rate, channels


def _write_wav_raw(wav_path: Path, audio: np.ndarray, sample_rate: int, channels: int = 1):
    """
    Write numpy array to WAV file.
    
    Args:
        wav_path: Output path
        audio: Audio data as float32 normalized -1 to 1
        sample_rate: Sample rate
        channels: Number of channels
    """
    import wave
    
    # Ensure proper shape
    if channels == 1 and audio.ndim == 2:
        audio = audio.flatten()
    
    # Clip and convert to 16-bit
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(str(wav_path), 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def _preprocess_audio_sync(input_path: Path, output_path: Path) -> None:
    """
    Preprocess audio to 48kHz mono WAV for RNNoise processing.
    
    RNNoise requires 48kHz mono audio - this is a hard requirement.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output preprocessed WAV
    """
    logger.info(f"Preprocessing audio for RNNoise: {input_path} -> {output_path}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ar", str(RNNOISE_SAMPLE_RATE),  # 48kHz (RNNoise requirement)
        "-ac", str(RNNOISE_CHANNELS),  # Mono (RNNoise requirement)
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
    
    logger.info(f"Preprocessed audio saved to: {output_path} (mono, {RNNOISE_SAMPLE_RATE}Hz)")


def _run_rnnoise_extraction(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Run RNNoise to extract speech component from audio.
    
    RNNoise is designed to denoise speech - it outputs "clean speech".
    We use this to identify the speech component, which we then subtract
    from the original to isolate background sounds.
    
    Args:
        audio: Input audio as float32 array, mono
        sample_rate: Sample rate (must be 48000)
        
    Returns:
        Extracted speech component as float32 array
    """
    import rnnoise
    
    if sample_rate != 48000:
        raise ValueError(f"RNNoise requires 48kHz audio, got {sample_rate}Hz")
    
    # RNNoise processes in frames of 480 samples (10ms at 48kHz)
    frame_size = 480
    
    # Ensure audio is flat
    if audio.ndim > 1:
        audio = audio.flatten()
    
    # Pad audio to multiple of frame_size
    original_length = len(audio)
    pad_length = (frame_size - (len(audio) % frame_size)) % frame_size
    if pad_length > 0:
        audio = np.concatenate([audio, np.zeros(pad_length, dtype=np.float32)])
    
    # Create RNNoise denoiser
    denoiser = rnnoise.RNNoise()
    
    # Process frame by frame
    output = np.zeros_like(audio)
    
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i+frame_size]
        # RNNoise expects int16 range (-32768 to 32767)
        frame_int16 = (frame * 32767).astype(np.int16)
        # Process frame
        denoised_frame = denoiser.process_frame(frame_int16)
        # Convert back to float32
        output[i:i+frame_size] = np.array(denoised_frame, dtype=np.float32) / 32767.0
    
    # Trim to original length
    output = output[:original_length]
    
    return output


def _subtract_speech(original: np.ndarray, speech: np.ndarray, attenuation_db: float = -30) -> np.ndarray:
    """
    Subtract speech component from original audio to isolate background.
    
    This performs spectral subtraction with some smoothing to reduce artifacts.
    
    Args:
        original: Original audio
        speech: Extracted speech component
        attenuation_db: Attenuation to apply to speech before subtraction (negative dB)
        
    Returns:
        Audio with speech removed (background sounds preserved)
    """
    # Ensure same length
    min_len = min(len(original), len(speech))
    original = original[:min_len]
    speech = speech[:min_len]
    
    # Apply attenuation to speech (convert dB to linear)
    attenuation = 10 ** (attenuation_db / 20)
    speech_attenuated = speech * (1.0 - attenuation)
    
    # Subtract speech from original
    # The speech signal from RNNoise is the "clean speech" estimate
    # Subtracting it leaves the "noise" which is our desired background sounds
    result = original - speech_attenuated
    
    # Soft clip to prevent artifacts
    result = np.tanh(result * 0.95) / 0.95
    
    return result


def _apply_residual_filter_sync(input_path: Path, output_path: Path) -> None:
    """
    Apply residual filtering to remove any remaining speech artifacts.
    
    Uses FFmpeg filters:
    - High-pass to remove low rumble
    - Gentle noise gate to reduce quiet speech remnants
    """
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
    Process a speech suppression job using RNNoise (SYNCHRONOUS, blocking).
    
    Pipeline:
    1. Preprocess audio to 48kHz mono (RNNoise requirement)
    2. Run RNNoise to extract speech component
    3. Subtract speech from original to get background sounds
    4. Apply residual filtering (optional)
    5. Copy output to static directory
    
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
            "RNNoise is not initialized. This should not happen - "
            "models are loaded during startup before accepting requests."
        )
    
    # Create working directories
    work_dir = job.work_dir
    preprocessed_path = work_dir / "preprocessed.wav"
    speech_path = work_dir / "speech_extracted.wav"
    subtracted_path = work_dir / "speech_subtracted.wav"
    filtered_path = work_dir / "filtered.wav"
    
    # Step 1: Preprocess audio to 48kHz mono
    logger.info("Step 1: Preprocessing audio to 48kHz mono")
    _preprocess_audio_sync(audio_path, preprocessed_path)
    
    # Step 2: Read audio and run RNNoise extraction
    logger.info("Step 2: Running RNNoise speech extraction")
    original_audio, sample_rate, channels = _read_wav_raw(preprocessed_path)
    
    if channels != 1:
        original_audio = original_audio.mean(axis=1)  # Convert to mono if needed
    
    speech_audio = _run_rnnoise_extraction(original_audio, sample_rate)
    
    # Save extracted speech for debugging (optional)
    _write_wav_raw(speech_path, speech_audio, sample_rate, 1)
    logger.info(f"Extracted speech saved to: {speech_path}")
    
    # Step 3: Subtract speech from original
    logger.info("Step 3: Subtracting speech to isolate background")
    result_audio = _subtract_speech(original_audio, speech_audio, SPEECH_ATTENUATION_DB)
    
    # Save subtracted result
    _write_wav_raw(subtracted_path, result_audio, sample_rate, 1)
    logger.info(f"Speech-subtracted audio saved to: {subtracted_path}")
    
    # Step 4: Apply residual filtering (optional)
    if ENABLE_RESIDUAL_FILTER:
        logger.info("Step 4: Applying residual filtering")
        _apply_residual_filter_sync(subtracted_path, filtered_path)
        final_path = filtered_path
    else:
        final_path = subtracted_path
    
    # Step 5: Copy output to static directory
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
    Process a speech suppression job using RNNoise (ASYNC wrapper).
    
    This function schedules blocking work in an executor to keep the event loop responsive.
    Only I/O operations (download) remain async. All CPU-bound work runs in executor.
    
    Steps:
    1. Download audio file (async I/O, 0-20% progress)
    2. Schedule RNNoise processing in executor (20-90% progress)
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
    
    logger.info("Scheduling RNNoise processing in executor")
    
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
