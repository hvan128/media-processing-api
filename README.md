# Media Processing API

A self-hosted backend API for audio/video processing workflows, designed for n8n integration.

## Features

- **Speech-to-Text (STT)**: Transcribe audio files to SRT subtitles using faster-whisper
- **Speech Suppression**: Remove speech from audio using RNNoise (returns background sounds)
- **Audio Merge**: Combine multiple audio tracks into video using FFmpeg

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│  POST /stt        POST /separate       POST /merge          │
│  POST /job/{id}   GET /static/{file}                        │
├─────────────────────────────────────────────────────────────┤
│                      Job Manager                             │
│  • Sequential queue (concurrency=1)                          │
│  • Progress tracking                                         │
│  • Automatic cleanup                                         │
├─────────────────────────────────────────────────────────────┤
│  STT Service     Separation Service     Merge Service        │
│  (faster-whisper)    (RNNoise)            (FFmpeg)           │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.10+
- FFmpeg
- 2-4GB RAM (CPU-only operation)

## Quick Start

### Local Development

```bash
# Clone or download the repository
cd translated_api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build image
docker build -t media-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/data media-api

# With environment variables
docker run -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e WHISPER_MODEL=base \
  media-api
```

### Docker Compose

```yaml
version: '3.8'
services:
  media-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
    restart: unless-stopped
```

## API Reference

### Speech-to-Text

**POST /stt**

Transcribe audio to SRT subtitles.

```bash
curl -X POST http://localhost:8000/stt \
  -H "Content-Type: application/json" \
  -d '{
    "media_url": "https://example.com/audio.mp3",
    "language": "zh",
    "output": "srt"
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

### Speech Suppression

**POST /separate**

Remove speech from audio (returns background sounds/instrumental).

```bash
curl -X POST http://localhost:8000/separate \
  -H "Content-Type: application/json" \
  -d '{
    "media_url": "https://example.com/song.mp3"
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "pending"
}
```

### Audio Merge

**POST /merge**

Merge audio tracks into video.

```bash
curl -X POST http://localhost:8000/merge \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "file_url": "https://example.com/video.mp4",
        "type": "video"
      },
      {
        "file_url": "https://example.com/voiceover.wav",
        "type": "audio",
        "volume": 1.0
      },
      {
        "file_url": "https://example.com/background_music.mp3",
        "type": "audio",
        "volume": 0.3
      }
    ],
    "options": {
      "mute_original_audio": true
    }
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440002",
  "status": "pending"
}
```

### Job Status

**GET /job/{job_id}**

Poll job status and retrieve results.

```bash
curl http://localhost:8000/job/550e8400-e29b-41d4-a716-446655440000
```

Responses:

**Pending:**
```json
{
  "status": "pending"
}
```

**Running:**
```json
{
  "status": "running",
  "progress": 45
}
```

**Done:**
```json
{
  "status": "done",
  "result": {
    "file_url": "/static/550e8400-e29b-41d4-a716-446655440000.srt"
  }
}
```

**Error:**
```json
{
  "status": "error",
  "error_message": "Download failed: 404 Not Found"
}
```

### Download Result

**GET /static/{filename}**

Download the processed file.

```bash
curl -O http://localhost:8000/static/550e8400-e29b-41d4-a716-446655440000.srt
```

## n8n Integration

### Workflow Pattern

1. **HTTP Request Node** (POST): Create job
2. **Wait Node**: Poll interval (e.g., 5 seconds)
3. **HTTP Request Node** (GET): Check job status
4. **IF Node**: Check if status == "done"
5. **Loop back** to Wait if not done
6. **Continue** to use result.file_url

### Example n8n Configuration

**Create STT Job:**
```json
{
  "method": "POST",
  "url": "{{$env.MEDIA_API_URL}}/stt",
  "body": {
    "media_url": "{{$json.audio_url}}",
    "language": "zh",
    "output": "srt"
  }
}
```

**Poll Job Status:**
```json
{
  "method": "GET",
  "url": "{{$env.MEDIA_API_URL}}/job/{{$json.job_id}}"
}
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `WHISPER_MODEL` | `base` | Whisper model size |
| `DATA_DIR` | `/data` | Data directory for outputs |

## Performance Notes

- Jobs are processed sequentially (concurrency=1) to prevent OOM on limited RAM
- Whisper and RNNoise models are loaded once at startup
- Video streams are copied without re-encoding for speed
- Temp files are automatically cleaned up after 24 hours

## Project Structure

```
translated_api/
├── main.py              # FastAPI app, routes, lifecycle
├── job_manager.py       # Job queue and state management
├── downloader.py        # HTTP file downloading
├── stt_service.py       # Speech-to-text (faster-whisper)
├── separation_service.py # Speech suppression (RNNoise)
├── merge_service.py     # Audio-video merge (FFmpeg)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container configuration
└── README.md           # This file
```

## License

MIT
