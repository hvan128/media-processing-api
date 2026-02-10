"""
Audio-Video Merge Service Module
================================
Handles merging audio tracks into video using FFmpeg.

Design Decisions:
- Video stream is copied (no re-encoding) when no trim is needed
- When video trimming is required, re-encodes with libx264 (fast preset)
- Audio tracks are mixed using FFmpeg's amix filter
- Volume scaling applied per-track before mixing
- Original video audio can be included or muted
- Generates filter_complex automatically based on inputs
- Automatic duration matching: probes durations and applies atempo or
  video trim so the client never needs to know duration differences
- Duration modes: auto_match_video, always_speed, always_trim_video,
  keep_original (default: auto_match_video)
"""

import asyncio
import json
import logging
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from job_manager import Job, JobManager
from downloader import download_file

logger = logging.getLogger(__name__)

VALID_DURATION_MODES = frozenset({
    "auto_match_video",
    "always_speed",
    "always_trim_video",
    "keep_original",
})


@dataclass
class MediaInput:
    """Represents a media input for merging"""
    file_path: Path
    input_type: str  # 'video' or 'audio'
    volume: float = 1.0
    input_index: int = 0  # FFmpeg input index
    duration_mode: str = "auto_match_video"


async def probe_duration(path: Path) -> float:
    """
    Probe media file duration using ffprobe.

    Uses ffprobe to extract ``format.duration`` from the file.

    Args:
        path: Path to the media file

    Returns:
        Duration in seconds as float

    Raises:
        RuntimeError: If ffprobe fails or duration cannot be parsed
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path),
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error = stderr.decode() if stderr else "unknown error"
        raise RuntimeError(f"ffprobe failed for {path}: {error}")

    data = json.loads(stdout.decode())

    try:
        duration = float(data["format"]["duration"])
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Could not parse duration from ffprobe output for {path}"
        ) from exc

    logger.info("Probed duration for %s: %.3fs", path.name, duration)
    return duration


def resolve_duration_action(
    mode: str,
    audio_duration: float,
    video_duration: float,
    tolerance: float = 0.02,
) -> str:
    """
    Decide how to handle an audio/video duration mismatch.

    Args:
        mode: Duration matching mode (one of VALID_DURATION_MODES)
        audio_duration: Duration of the audio track in seconds
        video_duration: Duration of the video in seconds
        tolerance: Ratio tolerance for considering durations equal

    Returns:
        One of: ``"speed_audio"``, ``"trim_video"``, ``"none"``
    """
    if mode == "keep_original":
        return "none"

    if mode == "always_speed":
        return "speed_audio"

    if mode == "always_trim_video":
        return "trim_video"

    if mode == "auto_match_video":
        if video_duration <= 0:
            logger.warning(
                "Video duration is zero or negative, skipping duration matching"
            )
            return "none"

        ratio = audio_duration / video_duration
        logger.info(
            "Duration ratio: %.4f (audio=%.3fs, video=%.3fs)",
            ratio, audio_duration, video_duration,
        )

        if abs(ratio - 1) < tolerance:
            logger.info(
                "Durations within tolerance (%.2f%%), no adjustment needed",
                tolerance * 100,
            )
            return "none"

        if ratio > 1:
            logger.info("Audio longer than video, will speed up audio")
            return "speed_audio"
        else:
            logger.info("Audio shorter than video, will trim video")
            return "trim_video"

    logger.warning("Unknown duration_mode '%s', defaulting to 'none'", mode)
    return "none"


def build_atempo_chain(factor: float) -> str:
    """
    Build an FFmpeg atempo filter chain for the given speed factor.

    FFmpeg's atempo filter only accepts values between 0.5 and 2.0.
    For factors outside this range, multiple atempo filters are chained.

    Examples::

        factor=4.0  -> "atempo=2.0,atempo=2.0"
        factor=0.25 -> "atempo=0.5,atempo=0.5"
        factor=1.5  -> "atempo=1.5000"

    Args:
        factor: Speed multiplication factor (must be positive)

    Returns:
        Comma-separated atempo filter chain string
    """
    if factor <= 0:
        raise ValueError(f"atempo factor must be positive, got {factor}")

    parts: list[str] = []
    remaining = factor

    # Handle factors above 2.0 by chaining max-rate stages
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0

    # Handle factors below 0.5 by chaining min-rate stages
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5

    # Final stage for the remainder within [0.5, 2.0]
    parts.append(f"atempo={remaining:.4f}")

    chain = ",".join(parts)
    logger.debug("Built atempo chain for factor %.4f: %s", factor, chain)
    return chain


def build_ffmpeg_command(
    video_input: MediaInput,
    audio_inputs: list[MediaInput],
    output_path: Path,
    mute_original_audio: bool = True,
    audio_tempo_filters: Optional[dict[int, str]] = None,
    video_trim_duration: Optional[float] = None,
) -> list[str]:
    """
    Build FFmpeg command for merging audio into video.

    Strategy:
    1. Map video stream (copy if no trim, re-encode if trimming)
    2. Apply volume and optional atempo filters to each audio input
    3. Mix all audio tracks using amix
    4. If not muting original, include video's audio in the mix
    5. If video_trim_duration is set, apply trim filter to video stream

    Args:
        video_input: Video file with input index
        audio_inputs: List of audio files to merge
        output_path: Path for output file
        mute_original_audio: Whether to discard original video audio
        audio_tempo_filters: Map of FFmpeg input index to atempo chain string
        video_trim_duration: If set, trim video to this duration in seconds

    Returns:
        FFmpeg command as list of arguments
    """
    if audio_tempo_filters is None:
        audio_tempo_filters = {}

    cmd = ["ffmpeg", "-y"]  # -y to overwrite output

    # Add all inputs
    # Video is always first input
    cmd.extend(["-i", str(video_input.file_path)])

    # Add audio inputs
    for audio in audio_inputs:
        cmd.extend(["-i", str(audio.file_path)])

    # Build filter_complex for audio mixing
    filter_parts: list[str] = []
    mix_inputs: list[str] = []

    # Current filter output label counter
    label_idx = 0

    # Video trim filter (if needed)
    needs_video_filter = video_trim_duration is not None
    if needs_video_filter:
        filter_parts.append(
            f"[0:v]trim=duration={video_trim_duration:.6f},"
            f"setpts=PTS-STARTPTS[vout]"
        )
        logger.info(
            "Adding video trim filter: duration=%.3fs", video_trim_duration
        )

    # Handle original video audio if not muted
    if not mute_original_audio:
        # Extract audio from video and apply volume (default 1.0)
        # Video is input 0, audio stream is 0:a
        label = f"a{label_idx}"
        filter_parts.append(f"[0:a]volume=1.0[{label}]")
        mix_inputs.append(f"[{label}]")
        label_idx += 1

    # Process each audio input with volume scaling and optional tempo adjustment
    for i, audio in enumerate(audio_inputs):
        # Audio inputs start at index 1 (video is 0)
        input_idx = i + 1
        label = f"a{label_idx}"

        # Apply volume filter
        # Volume values: 1.0 = original, 0.5 = half, 2.0 = double
        volume = max(0.0, min(2.0, audio.volume))  # Clamp to reasonable range

        # Build audio filter chain: volume [+ atempo]
        audio_filter = f"[{input_idx}:a]volume={volume}"

        if input_idx in audio_tempo_filters:
            audio_filter += f",{audio_tempo_filters[input_idx]}"
            logger.info(
                "Audio input %d: volume=%.2f, tempo=%s",
                input_idx, volume, audio_tempo_filters[input_idx],
            )

        audio_filter += f"[{label}]"
        filter_parts.append(audio_filter)
        mix_inputs.append(f"[{label}]")
        label_idx += 1

    # Build amix filter
    # amix mixes multiple audio streams into one
    # duration=first: output duration matches first input (video)
    # dropout_transition: fade out when streams end
    num_inputs = len(mix_inputs)

    if num_inputs == 1:
        # Single audio input – no mixing needed.
        # Replace the output label of the last audio filter with [aout]
        filter_parts[-1] = filter_parts[-1].rsplit("[", 1)[0] + "[aout]"
        filter_complex = ";".join(filter_parts)
    else:
        # Multiple inputs – use amix
        mix_input_labels = "".join(mix_inputs)
        filter_parts.append(
            f"{mix_input_labels}amix=inputs={num_inputs}"
            f":duration=first:dropout_transition=2[aout]"
        )
        filter_complex = ";".join(filter_parts)

    cmd.extend(["-filter_complex", filter_complex])

    # Map video stream
    if needs_video_filter:
        # Video was filtered – must re-encode
        cmd.extend([
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        ])
    else:
        # No video filter – copy stream for speed
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
    1. Download all input files (0-30% progress)
    2. Probe durations and resolve duration actions (30-40% progress)
    3. Build and run FFmpeg command (40-95% progress)
    4. Move output to static directory (95-100% progress)

    Args:
        job: Job instance with params:
            - inputs: List of {file_url, type, volume?, duration_mode?}
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
        raise ValueError(
            f"Exactly one video input required, got {len(video_inputs)}"
        )

    if len(audio_inputs) == 0 and mute_original_audio:
        raise ValueError(
            "At least one audio input required when muting original audio"
        )

    # ------------------------------------------------------------------
    # Step 1: Download all files (0-30%)
    # ------------------------------------------------------------------
    await manager.update_progress(job.job_id, 5)

    # Download video
    video_url = video_inputs[0]["file_url"]
    video_path = await download_file(
        url=video_url,
        dest_dir=job.work_dir,
        filename="video",
    )
    video_media = MediaInput(
        file_path=video_path,
        input_type="video",
        input_index=0,
    )

    await manager.update_progress(job.job_id, 15)

    # Download audio files
    audio_media_list: list[MediaInput] = []
    progress_per_audio = 15 / max(len(audio_inputs), 1)

    for i, audio_input in enumerate(audio_inputs):
        audio_url = audio_input["file_url"]
        volume = audio_input.get("volume", 1.0)

        # Parse duration_mode with backward-compatible default
        raw_mode = audio_input.get("duration_mode", "auto_match_video")
        if raw_mode not in VALID_DURATION_MODES:
            logger.warning(
                "Invalid duration_mode '%s' for audio %d, "
                "falling back to 'auto_match_video'",
                raw_mode, i,
            )
            raw_mode = "auto_match_video"

        audio_path = await download_file(
            url=audio_url,
            dest_dir=job.work_dir,
            filename=f"audio_{i}",
        )

        audio_media = MediaInput(
            file_path=audio_path,
            input_type="audio",
            volume=volume,
            input_index=i + 1,
            duration_mode=raw_mode,
        )
        audio_media_list.append(audio_media)

        progress = 15 + int((i + 1) * progress_per_audio)
        await manager.update_progress(job.job_id, progress)

    await manager.update_progress(job.job_id, 30)

    # ------------------------------------------------------------------
    # Step 2: Probe durations & resolve duration actions (30-40%)
    # ------------------------------------------------------------------
    logger.info("Probing media durations …")

    probe_tasks = [probe_duration(video_path)]
    for am in audio_media_list:
        probe_tasks.append(probe_duration(am.file_path))

    durations = await asyncio.gather(*probe_tasks)

    video_duration: float = durations[0]
    audio_durations: list[float] = list(durations[1:])

    logger.info(
        "Video duration: %.3fs | Audio durations: %s",
        video_duration,
        ", ".join(f"{d:.3f}s" for d in audio_durations),
    )

    # Resolve per-track duration actions
    audio_tempo_filters: dict[int, str] = {}
    trim_durations: list[float] = []

    for audio_media, audio_dur in zip(audio_media_list, audio_durations):
        action = resolve_duration_action(
            mode=audio_media.duration_mode,
            audio_duration=audio_dur,
            video_duration=video_duration,
        )

        input_idx = audio_media.input_index

        if action == "speed_audio":
            factor = audio_dur / video_duration
            atempo_chain = build_atempo_chain(factor)
            audio_tempo_filters[input_idx] = atempo_chain
            logger.info(
                "Audio %d: speed_audio – factor=%.4f, chain=%s",
                input_idx, factor, atempo_chain,
            )
        elif action == "trim_video":
            trim_durations.append(audio_dur)
            logger.info(
                "Audio %d: trim_video to %.3fs", input_idx, audio_dur,
            )
        else:
            logger.info("Audio %d: no duration adjustment needed", input_idx)

    video_trim_duration: Optional[float] = (
        min(trim_durations) if trim_durations else None
    )
    if video_trim_duration is not None:
        logger.info("Global video trim duration: %.3fs", video_trim_duration)

    has_duration_adjustments = bool(audio_tempo_filters) or video_trim_duration is not None

    await manager.update_progress(job.job_id, 40)

    # ------------------------------------------------------------------
    # Step 3: Build and run FFmpeg (40-95%)
    # ------------------------------------------------------------------
    output_filename = f"{job.job_id}_merged.mp4"
    temp_output = job.work_dir / output_filename

    # Choose command strategy
    use_simple = (
        len(audio_media_list) == 1
        and mute_original_audio
        and abs(audio_media_list[0].volume - 1.0) < 0.01
        and not has_duration_adjustments
    )

    if use_simple:
        # Simple case: single audio at normal volume, muting original, no
        # duration adjustments
        cmd = build_simple_ffmpeg_command(
            video_media, audio_media_list, temp_output, mute_original_audio,
        )
    else:
        # Complex case: use filter_complex (with optional atempo / trim)
        cmd = build_ffmpeg_command(
            video_media,
            audio_media_list,
            temp_output,
            mute_original_audio,
            audio_tempo_filters=audio_tempo_filters,
            video_trim_duration=video_trim_duration,
        )

    logger.debug("FFmpeg command: %s", " ".join(cmd))
    await manager.update_progress(job.job_id, 45)

    # Run FFmpeg
    returncode, stdout, stderr = await run_ffmpeg(cmd)

    if returncode != 0:
        # Extract useful error message
        error_lines = stderr.strip().split("\n")[-5:]  # Last 5 lines
        error_msg = "\n".join(error_lines)
        raise RuntimeError(f"FFmpeg failed (code {returncode}): {error_msg}")

    await manager.update_progress(job.job_id, 95)

    # ------------------------------------------------------------------
    # Step 4: Move output to static directory (95-100%)
    # ------------------------------------------------------------------
    output_path = manager.output_dir / output_filename
    shutil.move(temp_output, output_path)

    await manager.update_progress(job.job_id, 100)

    # Cleanup working directory
    manager.cleanup_job_work_dir(job)

    return {
        "file_url": f"/static/{output_filename}",
    }
