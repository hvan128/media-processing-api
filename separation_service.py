"""
Vocal Separation Service Module
===============================
Handles vocal separation using Spleeter, optimized for speech + industrial sounds.

Design Decisions:
- Uses Spleeter 2stems model (vocals/accompaniment)
- CPU-only operation, optimized for quality over speed
- Only returns accompaniment track (no_vocals) as per requirements
- Uses Spleeter as PYTHON LIBRARY (not CLI) to avoid typer import issues

Audio Quality Optimizations for Industrial Sounds:
- 44.1kHz sample rate (CD quality) preserves high frequencies
  * Important for metal drilling, impacts, tool sounds which have high-frequency content
  * Trade-off: ~2.75x slower than 16kHz, but much better quality
- Keeps stereo if input is stereo (better spatial information)

Aggressive Speech Suppression (ENABLED):
- High-pass filter at 6kHz removes speech fundamentals (300-3000Hz)
- Notch filter at 5-7kHz removes sibilants, breathing sounds, ending sounds
  * These artifacts (s, sh, ch sounds, breathing) are particularly annoying
  * They occur in 4-8kHz range, so notch filter specifically targets them
- Tool sounds (drilling, metal impacts) have energy > 6-8kHz, so they survive
- Trade-off: Some tool sound quality in low/mid frequencies is sacrificed
  for maximum speech removal

Expected Performance:
- ~25-30 seconds for 1 minute audio on CPU (with 44.1kHz)
- Pre-downloads model with redirect handling to fix GitHub 302 issue

Limitations:
- Spleeter was trained on music datasets, not speech + industrial sounds
- Perfect separation is not possible, but quality is improved with above optimizations
"""

import shutil
import subprocess
import asyncio
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import httpx

from job_manager import Job, JobManager
from downloader import download_file

# Set up logging
logger = logging.getLogger(__name__)

# Spleeter configuration
MODEL_NAME = "spleeter:2stems"
# Target sample rate: 44100Hz (CD quality) to preserve high frequencies
# Important for industrial sounds (drilling, metal impacts) which have high-frequency content
# Trade-off: ~2.75x slower than 16kHz, but much better quality for non-music audio
TARGET_SAMPLE_RATE = 44100
# Keep stereo if input is stereo (better spatial information for tools)
# Mono is forced only if input is mono
KEEP_STEREO = True

# Vocals blend configuration.
# NOTE: For maximum speech removal we disable blending entirely (ratio = 0.0).
# The blending pipeline is kept for potential future profiles but currently OFF.
VOCALS_BLEND_RATIO = 0.0
VOCALS_HIGHPASS_FREQ = 5000  # Only used if VOCALS_BLEND_RATIO > 0.0

# Aggressive speech suppression on final accompaniment:
# - Apply a strong high-pass filter + notch filter to remove speech artifacts
# - Human speech: 300-3000Hz (fundamental), but sibilants/breathing can reach 4-8kHz
# - Strategy: High-pass at 6kHz + notch filter at 5-7kHz to remove sibilants/breathing
# - Tool sounds (drilling, metal impacts) have significant energy > 6-8kHz
ENABLE_SPEECH_SUPPRESSION = True
SPEECH_SUPPRESSION_HIGHPASS_FREQ = 6000  # Hz; high-pass to remove speech fundamentals
SPEECH_SUPPRESSION_NOTCH_CENTER = 6000  # Hz; center frequency for notch filter
SPEECH_SUPPRESSION_NOTCH_WIDTH = 2000  # Hz; width of notch (removes 5-7kHz range)

# Model download configuration
SPLEETER_MODEL_URL = "https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz"
# Use MODEL_PATH env var if set (Docker), otherwise use home directory
_model_base = Path(os.environ.get("MODEL_PATH", str(Path.home() / "pretrained_models")))
SPLEETER_MODEL_DIR = _model_base / "2stems"

# Global separator instance
_separator = None
_model_loaded = False
_model_loading_error = None


def _download_spleeter_model():
    """
    Download Spleeter 2stems model manually with redirect support.
    
    GitHub releases return 302 redirects that Spleeter's internal downloader
    doesn't handle properly. This function downloads the model using httpx
    which correctly follows redirects.
    """
    # Check if model already exists
    model_json = SPLEETER_MODEL_DIR / "checkpoint"
    if model_json.exists():
        print(f"Spleeter model already exists at {SPLEETER_MODEL_DIR}")
        return
    
    print(f"Downloading Spleeter 2stems model from GitHub...")
    print(f"URL: {SPLEETER_MODEL_URL}")
    
    # Create model directory
    SPLEETER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download with redirect support
    with httpx.Client(follow_redirects=True, timeout=300.0) as client:
        response = client.get(SPLEETER_MODEL_URL)
        response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
    
    print(f"Downloaded model archive, extracting...")
    
    # Extract tar.gz
    try:
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=SPLEETER_MODEL_DIR)
        print(f"Model extracted to {SPLEETER_MODEL_DIR}")
    finally:
        # Clean up temp file
        os.unlink(tmp_path)
    
    # Verify extraction
    if not model_json.exists():
        # List extracted files for debugging
        extracted = list(SPLEETER_MODEL_DIR.iterdir())
        raise RuntimeError(
            f"Model extraction failed. Expected {model_json}. "
            f"Found: {[f.name for f in extracted]}"
        )
    
    print("Spleeter model downloaded and extracted successfully")


def load_model():
    """
    Pre-load Spleeter model.
    
    Loads Spleeter separator at startup to avoid delays during job processing.
    Uses Python library directly (not CLI) to avoid typer import issues.
    Pre-downloads model with proper redirect handling.
    """
    global _separator, _model_loaded, _model_loading_error
    
    if _model_loaded:
        return
    
    print(f"Initializing Spleeter ({MODEL_NAME}) for CPU...")
    
    try:
        # Set TensorFlow to CPU-only mode before importing
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
        
        # Set model directory for Spleeter (parent of 2stems folder)
        os.environ['MODEL_PATH'] = str(SPLEETER_MODEL_DIR.parent)
        
        # Pre-download model with redirect support (fixes GitHub 302 issue)
        _download_spleeter_model()
        
        # Import spleeter library directly (bypasses CLI and typer)
        from spleeter.separator import Separator
        
        # Create separator instance - model is already downloaded
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
    Preprocess audio to 44.1kHz WAV using FFmpeg (SYNCHRONOUS).
    
    Optimized for speech + industrial sounds (drilling, metal impacts):
    - 44.1kHz sample rate preserves high frequencies (important for metal sounds)
    - Keeps stereo if input is stereo (better spatial information)
    - Only converts to mono if input is mono
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output preprocessed WAV
    """
    logger.info(f"Preprocessing audio: {input_path} -> {output_path}")
    
    # Check if input is stereo by probing with ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path)
    ]
    probe_result = subprocess.run(
        probe_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    
    # Determine channel configuration
    if probe_result.returncode == 0:
        channels = probe_result.stdout.decode().strip()
        is_stereo = channels == "2" and KEEP_STEREO
    else:
        # Fallback: assume mono if probe fails
        is_stereo = False
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ar", str(TARGET_SAMPLE_RATE),  # 44.1kHz for high-frequency preservation
        "-acodec", "pcm_s16le",  # 16-bit PCM
    ]
    
    # Only force mono if input is mono or KEEP_STEREO is False
    if not is_stereo:
        cmd.extend(["-ac", "1"])  # Mono
    
    cmd.append(str(output_path))
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.decode() if result.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg preprocessing failed: {error_msg}")
    
    channel_info = "stereo" if is_stereo else "mono"
    logger.info(f"Preprocessed audio saved to: {output_path} ({channel_info}, {TARGET_SAMPLE_RATE}Hz)")


def _blend_audio_sync(
    accompaniment_path: Path,
    vocals_path: Path,
    output_path: Path,
    blend_ratio: float,
    highpass_freq: int = 5000,
) -> None:
    """
    Blend HIGH-FREQUENCY portion of vocals back into accompaniment using FFmpeg.

    NOTE: Currently disabled by setting VOCALS_BLEND_RATIO = 0.0 for maximum
    speech removal. The implementation is kept for future tuning profiles.
    """
    if blend_ratio <= 0.0:
        # No blending requested, just copy accompaniment.
        shutil.copy2(accompaniment_path, output_path)
        return

    logger.info(
        f"Blending {blend_ratio*100:.1f}% of vocals (high-pass >{highpass_freq}Hz) "
        f"back into accompaniment to recover tool sounds"
    )

    filter_complex = (
        f"[1:a]highpass=f={highpass_freq},volume={blend_ratio}[voc_filtered];"
        f"[0:a][voc_filtered]amix=inputs=2:duration=first:dropout_transition=0"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(accompaniment_path),
        "-i",
        str(vocals_path),
        "-filter_complex",
        filter_complex,
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        error_msg = result.stderr.decode() if result.stderr else "Unknown error"
        logger.warning(
            f"Audio blending failed: {error_msg}, using original accompaniment"
        )
        shutil.copy2(accompaniment_path, output_path)
    else:
        logger.info(f"Blended audio saved to: {output_path}")


def _suppress_speech_sync(input_path: Path, output_path: Path) -> None:
    """
    Apply aggressive speech suppression using high-pass + notch filters.

    This removes:
    - Speech fundamentals (300-3000Hz) via high-pass filter
    - Sibilants, breathing sounds, ending sounds (4-8kHz) via notch filter
    - These artifacts are particularly annoying when mixed with tool sounds

    Strategy:
    - High-pass at 6kHz removes speech fundamentals
    - Notch filter at 5-7kHz removes sibilants (s, sh, ch) and breathing sounds
    - Tool sounds (drilling, metal impacts) have energy > 6-8kHz, so they survive
    """
    logger.info(
        f"Applying aggressive speech suppression: "
        f"high-pass at {SPEECH_SUPPRESSION_HIGHPASS_FREQ}Hz, "
        f"notch at {SPEECH_SUPPRESSION_NOTCH_CENTER}Hz "
        f"(width: {SPEECH_SUPPRESSION_NOTCH_WIDTH}Hz)"
    )

    # FFmpeg filter chain:
    # 1. High-pass filter to remove speech fundamentals (< 6kHz)
    # 2. Notch filter (equalizer with negative gain) to remove sibilants/breathing (5-7kHz)
    #    Using equalizer with negative gain as a notch filter
    notch_low = SPEECH_SUPPRESSION_NOTCH_CENTER - SPEECH_SUPPRESSION_NOTCH_WIDTH // 2
    notch_high = SPEECH_SUPPRESSION_NOTCH_CENTER + SPEECH_SUPPRESSION_NOTCH_WIDTH // 2
    
    # Create filter chain: highpass + equalizer (notch) at 5-7kHz
    # Equalizer: center=6000Hz, bandwidth=2000Hz, gain=-20dB (strong reduction)
    filter_chain = (
        f"highpass=f={SPEECH_SUPPRESSION_HIGHPASS_FREQ},"
        f"equalizer=f={SPEECH_SUPPRESSION_NOTCH_CENTER}:width_type=h:width={SPEECH_SUPPRESSION_NOTCH_WIDTH}:g=-20"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        filter_chain,
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        error_msg = result.stderr.decode() if result.stderr else "Unknown error"
        logger.warning(
            f"Speech suppression filter failed: {error_msg}, using original accompaniment"
        )
        shutil.copy2(input_path, output_path)
    else:
        logger.info(f"Speech-suppressed accompaniment saved to: {output_path}")


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
    
    # Spleeter outputs to: output_dir/audio_stem/accompaniment.wav and vocals.wav
    audio_stem = audio_path.stem
    stem_dir = output_dir / audio_stem
    accompaniment_path = stem_dir / "accompaniment.wav"
    vocals_path = stem_dir / "vocals.wav"
    
    if not accompaniment_path.exists():
        # List files in output directory for debugging
        output_files = list(stem_dir.iterdir()) if stem_dir.exists() else []
        raise RuntimeError(
            f"Spleeter output not found: {accompaniment_path}. "
            f"Available files: {[f.name for f in output_files]}"
        )
    
    logger.info(f"Spleeter separation completed: {accompaniment_path}")
    
    # Blend HIGH-FREQUENCY vocals back into accompaniment to recover misclassified tool sounds
    # NOTE: Currently disabled via VOCALS_BLEND_RATIO = 0.0 for maximum speech removal.
    if VOCALS_BLEND_RATIO > 0.0 and vocals_path.exists():
        blended_path = stem_dir / "accompaniment_blended.wav"
        _blend_audio_sync(
            accompaniment_path, 
            vocals_path, 
            blended_path, 
            VOCALS_BLEND_RATIO,
            VOCALS_HIGHPASS_FREQ
        )
        return blended_path
    
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

    # Step 3: Aggressive speech suppression (optional)
    if ENABLE_SPEECH_SUPPRESSION:
        suppressed_path = work_dir / "accompaniment_suppressed.wav"
        _suppress_speech_sync(accompaniment_path, suppressed_path)
        final_path = suppressed_path
    else:
        final_path = accompaniment_path

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
