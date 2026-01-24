"""
Speech-to-Text Service Module
=============================
Handles transcription using faster-whisper.

Design Decisions:
- Model loaded once at startup and reused for all requests
- Uses 'base' model with int8 quantization for CPU efficiency
- VAD filter enabled to skip silence and reduce processing time
- Outputs SRT format for subtitle compatibility
"""

import shutil
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from job_manager import Job, JobManager
from downloader import download_file


# Model configuration optimized for CPU-only VPS
MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # int8 for lower memory usage on CPU

# Global model instance - loaded once at startup
_model: Optional[WhisperModel] = None


def load_model():
    """
    Load the Whisper model into memory.
    
    Called once at application startup.
    Model is reused across all requests to avoid repeated loading overhead.
    """
    global _model
    if _model is None:
        print(f"Loading faster-whisper model: {MODEL_SIZE} ({COMPUTE_TYPE})")
        _model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("Model loaded successfully")
    return _model


def get_model() -> WhisperModel:
    """Get the loaded model instance"""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format: HH:MM:SS,mmm
    
    SRT requires comma as decimal separator (not period).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list) -> str:
    """
    Convert Whisper segments to SRT format.
    
    SRT format:
    1
    00:00:00,000 --> 00:00:02,500
    First subtitle text
    
    2
    00:00:02,500 --> 00:00:05,000
    Second subtitle text
    """
    srt_lines = []
    
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between entries
    
    return "\n".join(srt_lines)


async def process_stt(job: Job, manager: JobManager) -> dict:
    """
    Process a speech-to-text job.
    
    Steps:
    1. Download audio file (0-20% progress)
    2. Run transcription (20-90% progress)
    3. Generate SRT output (90-100% progress)
    
    Args:
        job: Job instance with params:
            - media_url: URL to audio file
            - language: Language code (e.g., 'zh', 'en')
            - output: Output format ('srt')
        manager: JobManager for progress updates
    
    Returns:
        dict with 'file_url' pointing to the output SRT file
    """
    params = job.params
    media_url = params["media_url"]
    language = params.get("language", "zh")
    output_format = params.get("output", "srt")
    
    # Validate output format
    if output_format != "srt":
        raise ValueError(f"Unsupported output format: {output_format}. Only 'srt' is supported.")
    
    # Step 1: Download audio file
    await manager.update_progress(job.job_id, 5)
    
    audio_path = await download_file(
        url=media_url,
        dest_dir=job.work_dir,
        filename="audio"
    )
    
    await manager.update_progress(job.job_id, 20)
    
    # Step 2: Run transcription
    model = get_model()
    
    # Transcribe with VAD filter for better accuracy and speed
    segments_generator, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,  # Minimum silence to split segments
            speech_pad_ms=200  # Padding around speech
        )
    )
    
    # Collect segments with progress updates
    # Since we don't know total duration upfront, estimate progress
    segments = []
    for segment in segments_generator:
        segments.append(segment)
        # Update progress based on segment timing vs detected duration
        if info.duration > 0:
            progress = 20 + int((segment.end / info.duration) * 70)
            progress = min(90, progress)
            await manager.update_progress(job.job_id, progress)
    
    await manager.update_progress(job.job_id, 90)
    
    # Step 3: Generate SRT output
    srt_content = segments_to_srt(segments)
    
    # Save to output directory
    output_filename = f"{job.job_id}.srt"
    output_path = manager.output_dir / output_filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    await manager.update_progress(job.job_id, 100)
    
    # Cleanup working directory (audio file)
    manager.cleanup_job_work_dir(job)
    
    return {
        "file_url": f"/static/{output_filename}"
    }
