"""
Audio-Video Merge Service Module
================================
Handles merging audio tracks into video using FFmpeg.

Design Decisions:
- Video stream is always copied (no re-encoding) for speed
- Audio tracks are mixed using FFmpeg's amix filter
- Volume scaling applied per-track before mixing
- Original video audio can be included or muted
- Generates filter_complex automatically based on inputs
"""

import asyncio
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from job_manager import Job, JobManager
from downloader import download_file


@dataclass
class MediaInput:
    """Represents a media input for merging"""
    file_path: Path
    input_type: str  # 'video' or 'audio'
    volume: float = 1.0
    input_index: int = 0  # FFmpeg input index


def build_ffmpeg_command(
    video_input: MediaInput,
    audio_inputs: list[MediaInput],
    output_path: Path,
    mute_original_audio: bool = True
) -> list[str]:
    """
    Build FFmpeg command for merging audio into video.
    
    Strategy:
    1. Map video stream directly (copy, no re-encode)
    2. Apply volume filter to each audio input
    3. Mix all audio tracks using amix
    4. If not muting original, include video's audio in the mix
    
    Args:
        video_input: Video file with input index
        audio_inputs: List of audio files to merge
        output_path: Path for output file
        mute_original_audio: Whether to discard original video audio
    
    Returns:
        FFmpeg command as list of arguments
    """
    cmd = ["ffmpeg", "-y"]  # -y to overwrite output
    
    # Add all inputs
    # Video is always first input
    cmd.extend(["-i", str(video_input.file_path)])
    
    # Add audio inputs
    for audio in audio_inputs:
        cmd.extend(["-i", str(audio.file_path)])
    
    # Build filter_complex for audio mixing
    filter_parts = []
    mix_inputs = []
    
    # Current filter output label counter
    label_idx = 0
    
    # Handle original video audio if not muted
    if not mute_original_audio:
        # Extract audio from video and apply volume (default 1.0)
        # Video is input 0, audio stream is 0:a
        label = f"a{label_idx}"
        filter_parts.append(f"[0:a]volume=1.0[{label}]")
        mix_inputs.append(f"[{label}]")
        label_idx += 1
    
    # Process each audio input with volume scaling
    for i, audio in enumerate(audio_inputs):
        # Audio inputs start at index 1 (video is 0)
        input_idx = i + 1
        label = f"a{label_idx}"
        
        # Apply volume filter
        # Volume values: 1.0 = original, 0.5 = half, 2.0 = double
        volume = max(0.0, min(2.0, audio.volume))  # Clamp to reasonable range
        filter_parts.append(f"[{input_idx}:a]volume={volume}[{label}]")
        mix_inputs.append(f"[{label}]")
        label_idx += 1
    
    # Build amix filter
    # amix mixes multiple audio streams into one
    # duration=first: output duration matches first input (video)
    # dropout_transition: fade out when streams end
    num_inputs = len(mix_inputs)
    
    if num_inputs == 1:
        # Single audio input - no mixing needed, just use it directly
        filter_complex = filter_parts[0].replace(f"[a0]", "[aout]")
    else:
        # Multiple inputs - use amix
        mix_input_labels = "".join(mix_inputs)
        filter_parts.append(
            f"{mix_input_labels}amix=inputs={num_inputs}:duration=first:dropout_transition=2[aout]"
        )
        filter_complex = ";".join(filter_parts)
    
    cmd.extend(["-filter_complex", filter_complex])
    
    # Map video stream (copy, no re-encode for speed)
    cmd.extend(["-map", "0:v", "-c:v", "copy"])
    
    # Map mixed audio output
    cmd.extend(["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"])
    
    # Output file
    cmd.append(str(output_path))
    
    return cmd


def build_simple_ffmpeg_command(
    video_input: MediaInput,
    audio_inputs: list[MediaInput],
    output_path: Path,
    mute_original_audio: bool = True
) -> list[str]:
    """
    Build a simpler FFmpeg command for single audio replacement.
    
    Used when there's exactly one audio input and we're muting original.
    This is more efficient than using filter_complex.
    """
    if len(audio_inputs) != 1 or not mute_original_audio:
        raise ValueError("Simple command only works with single audio and muted original")
    
    audio = audio_inputs[0]
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_input.file_path),
        "-i", str(audio.file_path),
        "-map", "0:v",  # Video from first input
        "-map", "1:a",  # Audio from second input
        "-c:v", "copy",  # Copy video stream
        "-c:a", "aac", "-b:a", "192k",  # Encode audio as AAC
        "-shortest",  # Cut to shortest stream
        str(output_path)
    ]
    
    # Apply volume if not 1.0
    if abs(audio.volume - 1.0) > 0.01:
        # Insert volume filter
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_input.file_path),
            "-i", str(audio.file_path),
            "-map", "0:v",
            "-filter_complex", f"[1:a]volume={audio.volume}[aout]",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path)
        ]
    
    return cmd


async def run_ffmpeg(cmd: list[str]) -> tuple[int, str, str]:
    """
    Run FFmpeg command asynchronously.
    
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    return (
        process.returncode,
        stdout.decode() if stdout else "",
        stderr.decode() if stderr else ""
    )


def parse_ffmpeg_progress(stderr: str, duration: float) -> int:
    """
    Parse FFmpeg stderr output to estimate progress.
    
    FFmpeg outputs time= HH:MM:SS.ms in stderr.
    We use this to calculate progress percentage.
    """
    # Find all time= matches
    matches = re.findall(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})", stderr)
    
    if not matches:
        return 0
    
    # Get the last (most recent) time
    last = matches[-1]
    hours, minutes, seconds, centis = map(int, last)
    current_time = hours * 3600 + minutes * 60 + seconds + centis / 100
    
    if duration > 0:
        progress = int((current_time / duration) * 100)
        return min(100, max(0, progress))
    
    return 0


async def process_merge(job: Job, manager: JobManager) -> dict:
    """
    Process an audio-video merge job.
    
    Steps:
    1. Download all input files (0-40% progress)
    2. Build and run FFmpeg command (40-95% progress)
    3. Move output to static directory (95-100% progress)
    
    Args:
        job: Job instance with params:
            - inputs: List of {file_url, type, volume?}
            - options: {mute_original_audio?}
        manager: JobManager for progress updates
    
    Returns:
        dict with 'file_url' pointing to the merged video file
    """
    params = job.params
    inputs = params["inputs"]
    options = params.get("options", {})
    mute_original_audio = options.get("mute_original_audio", True)
    
    # Validate inputs
    video_inputs = [i for i in inputs if i.get("type") == "video"]
    audio_inputs = [i for i in inputs if i.get("type") == "audio"]
    
    if len(video_inputs) != 1:
        raise ValueError(f"Exactly one video input required, got {len(video_inputs)}")
    
    if len(audio_inputs) == 0 and mute_original_audio:
        raise ValueError("At least one audio input required when muting original audio")
    
    # Step 1: Download all files
    await manager.update_progress(job.job_id, 5)
    
    # Download video
    video_url = video_inputs[0]["file_url"]
    video_path = await download_file(
        url=video_url,
        dest_dir=job.work_dir,
        filename="video"
    )
    video_media = MediaInput(
        file_path=video_path,
        input_type="video",
        input_index=0
    )
    
    await manager.update_progress(job.job_id, 15)
    
    # Download audio files
    audio_media_list = []
    progress_per_audio = 25 / max(len(audio_inputs), 1)
    
    for i, audio_input in enumerate(audio_inputs):
        audio_url = audio_input["file_url"]
        volume = audio_input.get("volume", 1.0)
        
        audio_path = await download_file(
            url=audio_url,
            dest_dir=job.work_dir,
            filename=f"audio_{i}"
        )
        
        audio_media = MediaInput(
            file_path=audio_path,
            input_type="audio",
            volume=volume,
            input_index=i + 1
        )
        audio_media_list.append(audio_media)
        
        progress = 15 + int((i + 1) * progress_per_audio)
        await manager.update_progress(job.job_id, progress)
    
    await manager.update_progress(job.job_id, 40)
    
    # Step 2: Build and run FFmpeg
    output_filename = f"{job.job_id}_merged.mp4"
    temp_output = job.work_dir / output_filename
    
    # Choose command strategy
    if len(audio_media_list) == 1 and mute_original_audio and abs(audio_media_list[0].volume - 1.0) < 0.01:
        # Simple case: single audio at normal volume, muting original
        cmd = build_simple_ffmpeg_command(
            video_media, audio_media_list, temp_output, mute_original_audio
        )
    else:
        # Complex case: use filter_complex
        cmd = build_ffmpeg_command(
            video_media, audio_media_list, temp_output, mute_original_audio
        )
    
    await manager.update_progress(job.job_id, 45)
    
    # Run FFmpeg
    returncode, stdout, stderr = await run_ffmpeg(cmd)
    
    if returncode != 0:
        # Extract useful error message
        error_lines = stderr.strip().split("\n")[-5:]  # Last 5 lines
        error_msg = "\n".join(error_lines)
        raise RuntimeError(f"FFmpeg failed (code {returncode}): {error_msg}")
    
    await manager.update_progress(job.job_id, 95)
    
    # Step 3: Move output to static directory
    output_path = manager.output_dir / output_filename
    shutil.move(temp_output, output_path)
    
    await manager.update_progress(job.job_id, 100)
    
    # Cleanup working directory
    manager.cleanup_job_work_dir(job)
    
    return {
        "file_url": f"/static/{output_filename}"
    }
