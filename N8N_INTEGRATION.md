# n8n Integration Guide

This guide explains how to connect n8n to the Media Processing API.

## Connection Methods

There are two ways to connect n8n to the API, depending on how n8n is deployed:

### Method 1: n8n on Host (Recommended)

If n8n is running directly on the host (not in Docker), use **localhost**:

```
Base URL: http://127.0.0.1:9300
```

**Example HTTP Request Node:**
```json
{
  "method": "POST",
  "url": "http://127.0.0.1:9300/stt",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "media_url": "{{ $json.audio_url }}",
    "language": "zh",
    "output": "srt"
  }
}
```

### Method 2: n8n in Docker (Same Network)

If n8n is running in Docker and on the same network as the API, use the **service name**:

```
Base URL: http://media-processing-api:8000
```

**Important:** 
- The service name is `media-processing-api` (from docker-compose.yml)
- Use port `8000` (internal container port, not the exposed `9300`)
- Both containers must be on the same Docker network (`internal-net`)

**Example n8n docker-compose.yml:**
```yaml
services:
  n8n:
    image: n8nio/n8n
    container_name: n8n
    networks:
      - internal-net  # Same network as media-processing-api
    # ... other config

networks:
  internal-net:
    external: true  # Use the same external network
```

**Example HTTP Request Node:**
```json
{
  "method": "POST",
  "url": "http://media-processing-api:8000/stt",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "media_url": "{{ $json.audio_url }}",
    "language": "zh",
    "output": "srt"
  }
}
```

## Troubleshooting

### Error: "getaddrinfo ENOTFOUND media-processing-api"

This means n8n cannot resolve the hostname. Solutions:

1. **If n8n is on the host:** Use `http://127.0.0.1:9300` instead
2. **If n8n is in Docker:** 
   - Ensure both services are on the same network
   - Check network: `docker network inspect internal-net`
   - Verify service name matches: `docker ps | grep media-processing-api`

### Verify Network Connection

**From n8n container:**
```bash
# If n8n is in Docker
docker exec -it n8n ping media-processing-api

# Or test HTTP connection
docker exec -it n8n curl http://media-processing-api:8000/health
```

**From host:**
```bash
# Test localhost connection
curl http://127.0.0.1:9300/health
```

### Check Service Status

```bash
# Check if API is running
docker ps | grep media-processing-api

# Check API health
curl http://127.0.0.1:9300/health

# Check API logs
docker logs media-processing-api
```

## API Endpoints

All endpoints are available at the base URL:

- `POST /stt` - Speech-to-text
- `POST /separate` - Speech suppression
- `POST /merge` - Audio-video merge
- `GET /job/{job_id}` - Check job status
- `GET /static/{filename}` - Download result file
- `GET /health` - Health check

## Example n8n Workflow

1. **HTTP Request Node** (POST `/stt`): Create transcription job
2. **Wait Node**: Wait 5 seconds
3. **HTTP Request Node** (GET `/job/{job_id}`): Check status
4. **IF Node**: Check if `status == "done"`
5. **Loop back** to Wait if not done
6. **HTTP Request Node** (GET `/static/{filename}`): Download result

## Environment Variables

You can use n8n environment variables for the base URL:

```json
{
  "url": "{{ $env.MEDIA_API_URL }}/stt"
}
```

Set in n8n:
- `MEDIA_API_URL=http://127.0.0.1:9300` (if n8n on host)
- `MEDIA_API_URL=http://media-processing-api:8000` (if n8n in Docker)
