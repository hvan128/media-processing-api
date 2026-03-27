"""
Vocal Separation Service Module (Demucs)
=========================================
Separates audio into vocal and instrumental tracks using Facebook's Demucs model.

Optimized for GPU GTX 1050 3GB:
- Uses htdemucs model (best quality/performance ratio)
- Two-stems extraction (vocals + instrumental)
- Configurable segment size (default 6s) to avoid OOM
- Single shift for memory efficiency
- GPU auto-detection with CPU fallback
- Model loaded once at startup, reused across requests
- Audio hash caching to skip duplicate processing

Pipeline:
1. Download audio from URL
2. Compute audio hash (check cache)
3. Load & resample audio to model sample rate
4. Run Demucs separation (GPU preferred, CPU fallback)
5. Extract vocal and instrumental stems
6. Save as WAV to /output directory
7. Cache result by audio hash
"""

import hashlib
import shutil
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import soundfile as sf

from job_manager import Job, JobManager
from downloader import download_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("DEMUCS_MODEL", "htdemucs")
SEGMENT_SIZE = int(os.environ.get("DEMUCS_SEGMENT", "6"))
SHIFTS = int(os.environ.get("DEMUCS_SHIFTS", "1"))
OVERLAP = float(os.environ.get("DEMUCS_OVERLAP", "0.25"))
MAX_CACHE_ENTRIES = int(os.environ.get("DEMUCS_CACHE_SIZE", "100"))

# ---------------------------------------------------------------------------
# Global state (singleton model, cache)
# ---------------------------------------------------------------------------
_model = None
_device: Optional[torch.device] = None
_result_cache: dict[str, dict[str, Path]] = {}


def _detect_device() -> torch.device:
    """Pick CUDA when available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = props.total_mem / (1024 ** 3)
        logger.info("GPU detected: %s (%.1f GB)", props.name, gpu_mem_gb)
        return torch.device("cuda")
    logger.info("No GPU detected — using CPU")
    return torch.device("cpu")


def load_model():
    """
    Load the Demucs model once.  Called during application startup so that
    every subsequent request reuses the same model weights in memory.
    """
    global _model, _device

    if _model is not None:
        return

    from demucs.pretrained import get_model as _get_demucs_model

    _device = _detect_device()

    print(f"Loading Demucs model '{MODEL_NAME}' on {_device} ...")
    _model = _get_demucs_model(MODEL_NAME)
    _model.to(_device)
    _model.eval()

    print(
        f"Demucs ready — sources: {_model.sources}, "
        f"samplerate: {_model.samplerate}, segment: {SEGMENT_SIZE}s, "
        f"shifts: {SHIFTS}, device: {_device}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_audio_hash(file_path: Path) -> str:
    """Return hex MD5 of the raw file bytes (fast, good enough for caching)."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _evict_cache_if_needed():
    """Drop the oldest entry when the cache exceeds MAX_CACHE_ENTRIES."""
    while len(_result_cache) > MAX_CACHE_ENTRIES:
        oldest_key = next(iter(_result_cache))
        _result_cache.pop(oldest_key, None)


def _cache_files_exist(cached: dict[str, Path]) -> bool:
    """Return True only when both cached WAV files still exist on disk."""
    return (
        cached.get("vocal_path") is not None
        and cached["vocal_path"].exists()
        and cached.get("instrument_path") is not None
        and cached["instrument_path"].exists()
    )


# ---------------------------------------------------------------------------
# Core separation (synchronous — runs inside a ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _process_separation_sync(
    job: Job,
    manager: JobManager,
    audio_path: Path,
) -> dict:
    """
    Run Demucs htdemucs separation and write vocal + instrumental WAVs.

    This function is **blocking** and must be called via ``run_in_executor``.
    """
    from demucs.apply import apply_model

    job_id = job.job_id
    output_dir = manager.output_dir

    # ------------------------------------------------------------------
    # 1. Hash-based cache lookup
    # ------------------------------------------------------------------
    audio_hash = _compute_audio_hash(audio_path)

    if audio_hash in _result_cache and _cache_files_exist(_result_cache[audio_hash]):
        cached = _result_cache[audio_hash]
        vocal_out = output_dir / f"{job_id}_vocal.wav"
        instrument_out = output_dir / f"{job_id}_instrument.wav"
        shutil.copy2(cached["vocal_path"], vocal_out)
        shutil.copy2(cached["instrument_path"], instrument_out)
        logger.info("Cache hit for job %s (hash=%s)", job_id, audio_hash)
        manager.cleanup_job_work_dir(job)
        return {
            "vocal_url": f"/output/{job_id}_vocal.wav",
            "instrument_url": f"/output/{job_id}_instrument.wav",
        }

    # ------------------------------------------------------------------
    # 2. Load audio (via soundfile) & resample to model sample rate
    # ------------------------------------------------------------------
    logger.info("Loading audio: %s", audio_path)
    # Use soundfile for I/O to avoid TorchCodec-dependent backends
    audio_np, sr = sf.read(str(audio_path), always_2d=False, dtype="float32")

    if audio_np.ndim == 1:
        wav = torch.from_numpy(audio_np).unsqueeze(0)  # [1, samples]
    else:
        # soundfile returns [samples, channels] – transpose to [channels, samples]
        wav = torch.from_numpy(audio_np.T)

    if sr != _model.samplerate:
        logger.info("Resampling %d → %d Hz", sr, _model.samplerate)
        wav = torchaudio.functional.resample(wav, sr, _model.samplerate)

    # Demucs expects stereo (2-channel) input
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # Batch dimension: [1, channels, samples]
    wav = wav.unsqueeze(0).to(_device)

    # ------------------------------------------------------------------
    # 3. Run Demucs separation
    # ------------------------------------------------------------------
    logger.info(
        "Running Demucs (model=%s, segment=%ds, shifts=%d, device=%s)",
        MODEL_NAME, SEGMENT_SIZE, SHIFTS, _device,
    )

    with torch.no_grad():
        sources = apply_model(
            _model,
            wav,
            device=_device,
            segment=SEGMENT_SIZE,
            shifts=SHIFTS,
            overlap=OVERLAP,
            progress=False,
        )
    # sources shape: [batch=1, num_sources, channels, samples]

    # ------------------------------------------------------------------
    # 4. Extract vocal and instrumental stems
    # ------------------------------------------------------------------
    vocal_idx = _model.sources.index("vocals")
    vocals = sources[0, vocal_idx].cpu()

    instrumental = torch.zeros_like(vocals)
    for i, name in enumerate(_model.sources):
        if name != "vocals":
            instrumental += sources[0, i].cpu()

    # Free GPU memory early
    del sources, wav
    if _device is not None and _device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Save WAV outputs
    # ------------------------------------------------------------------
    vocal_out = output_dir / f"{job_id}_vocal.wav"
    instrument_out = output_dir / f"{job_id}_instrument.wav"

    # Save via soundfile to avoid TorchCodec-based I/O
    sf.write(str(vocal_out), vocals.cpu().numpy().T, _model.samplerate)
    sf.write(str(instrument_out), instrumental.cpu().numpy().T, _model.samplerate)

    logger.info("Saved: %s, %s", vocal_out, instrument_out)

    # ------------------------------------------------------------------
    # 6. Update cache
    # ------------------------------------------------------------------
    _evict_cache_if_needed()
    _result_cache[audio_hash] = {
        "vocal_path": vocal_out,
        "instrument_path": instrument_out,
    }

    # ------------------------------------------------------------------
    # 7. Cleanup temp working directory
    # ------------------------------------------------------------------
    manager.cleanup_job_work_dir(job)

    return {
        "vocal_url": f"/output/{job_id}_vocal.wav",
        "instrument_url": f"/output/{job_id}_instrument.wav",
    }


# ---------------------------------------------------------------------------
# Async entry point (called by JobManager worker)
# ---------------------------------------------------------------------------

async def process_separation(job: Job, manager: JobManager) -> dict:
    """
    Download the audio, run Demucs in a thread-pool executor, and return
    URLs for the separated vocal and instrumental tracks.

    Progress milestones:
        0-10%   downloading
       10-20%   loading & preprocessing audio
       20-90%   Demucs model inference
       90-100%  saving outputs & cleanup
    """
    media_url = job.params["media_url"]

    # --- Download --------------------------------------------------------
    await manager.update_progress(job.job_id, 5)
    logger.info("Downloading audio: %s", media_url)

    audio_path = await download_file(
        url=media_url,
        dest_dir=job.work_dir,
        filename="audio",
    )
    await manager.update_progress(job.job_id, 15)

    # --- Separation (CPU-bound → executor) -------------------------------
    await manager.update_progress(job.job_id, 20)
    logger.info("Scheduling Demucs separation in executor")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        _process_separation_sync,
        job,
        manager,
        audio_path,
    )

    await manager.update_progress(job.job_id, 100)
    logger.info("Separation job %s completed", job.job_id)

    return result
