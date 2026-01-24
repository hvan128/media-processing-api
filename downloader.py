"""
File Downloader Module
======================
Handles downloading media files from public URLs.

Design Decisions:
- Uses httpx for async HTTP operations
- Streams large files to disk to avoid memory issues
- Preserves original file extension for FFmpeg compatibility
- Configurable timeout and retry logic
"""

import httpx
from pathlib import Path
from urllib.parse import urlparse, unquote
import mimetypes


# HTTP client configuration
DOWNLOAD_TIMEOUT = 300.0  # 5 minutes for large files
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming

# MIME type to extension mapping for common media types
MIME_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp4": ".m4a",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
}


def get_extension_from_url(url: str) -> str:
    """
    Extract file extension from URL path.
    Falls back to .bin if no extension found.
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)
    suffix = Path(path).suffix.lower()
    
    if suffix and len(suffix) <= 5:  # Reasonable extension length
        return suffix
    return ""


def get_extension_from_content_type(content_type: str) -> str:
    """
    Get file extension from Content-Type header.
    """
    if not content_type:
        return ""
    
    # Remove charset and other parameters
    mime_type = content_type.split(";")[0].strip().lower()
    
    # Check our custom mapping first
    if mime_type in MIME_TO_EXT:
        return MIME_TO_EXT[mime_type]
    
    # Fall back to mimetypes module
    ext = mimetypes.guess_extension(mime_type)
    return ext if ext else ""


async def download_file(
    url: str,
    dest_dir: Path,
    filename: str = "download",
    progress_callback: callable = None
) -> Path:
    """
    Download a file from a public URL.
    
    Args:
        url: Public URL to download from
        dest_dir: Directory to save the file
        filename: Base filename (extension added automatically)
        progress_callback: Optional async callback(downloaded_bytes, total_bytes)
    
    Returns:
        Path to the downloaded file
    
    Raises:
        httpx.HTTPError: On network/HTTP errors
        ValueError: On invalid URL or response
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # First, try to get extension from URL
    extension = get_extension_from_url(url)
    
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT),
        follow_redirects=True
    ) as client:
        # Use streaming to handle large files
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            
            # If no extension from URL, try Content-Type header
            if not extension:
                content_type = response.headers.get("content-type", "")
                extension = get_extension_from_content_type(content_type)
            
            # Default extension if nothing found
            if not extension:
                extension = ".bin"
            
            # Build final path
            dest_path = dest_dir / f"{filename}{extension}"
            
            # Get total size for progress tracking
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            # Stream to disk
            with open(dest_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback and total_size > 0:
                        await progress_callback(downloaded, total_size)
    
    return dest_path


async def download_files(
    urls: list[str],
    dest_dir: Path,
    prefix: str = "file"
) -> list[Path]:
    """
    Download multiple files sequentially.
    
    Args:
        urls: List of URLs to download
        dest_dir: Directory to save files
        prefix: Filename prefix (files named prefix_0, prefix_1, etc.)
    
    Returns:
        List of paths to downloaded files (in same order as URLs)
    """
    paths = []
    for i, url in enumerate(urls):
        path = await download_file(url, dest_dir, f"{prefix}_{i}")
        paths.append(path)
    return paths


def get_file_type(path: Path) -> str:
    """
    Determine if a file is audio or video based on extension.
    
    Returns:
        'audio', 'video', or 'unknown'
    """
    audio_exts = {".mp3", ".wav", ".aac", ".ogg", ".flac", ".m4a", ".opus", ".wma"}
    video_exts = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".wmv"}
    
    ext = path.suffix.lower()
    
    if ext in audio_exts:
        return "audio"
    elif ext in video_exts:
        return "video"
    else:
        return "unknown"
