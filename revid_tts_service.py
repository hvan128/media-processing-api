"""
Revid TTS Service
=================
SRT-to-Speech via Revid API (https://api.revidapi.com/paid/srt-to-speech/merge).

- Submits subtitles to Revid, polls until completed, downloads audio to output_dir.
- API key rotation:
    - Tracks per-key state: active / exhausted (402) / invalid (401)
    - Selects next active key in round-robin order
- Response format aligned with project: job_id → GET /job/{job_id} → result.file_url.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import httpx

from job_manager import Job, JobManager

logger = logging.getLogger(__name__)

REVID_MERGE_URL = "https://api.revidapi.com/paid/srt-to-speech/merge"
REVID_GET_TASK_URL = "https://tts.revidapi.com/api/get"
REVID_POLL_INTERVAL = 2.0
REVID_POLL_TIMEOUT = 300.0  # 5 minutes
REVID_REQUEST_TIMEOUT = 60.0

CONFIG_DIR = Path("/data/config")
REVID_KEYS_FILE = CONFIG_DIR / "revid_api_keys.txt"
# Fallback for Windows / dev when /data doesn't exist
FALLBACK_CONFIG = Path(__file__).resolve().parent / "data" / "config" / "revid_api_keys.txt"


def _config_path() -> Path:
    if REVID_KEYS_FILE.exists():
        return REVID_KEYS_FILE
    return FALLBACK_CONFIG


def _state_path() -> Path:
    """Path to persisted key state JSON (same directory as keys file)."""
    return _config_path().with_name("revid_key_state.json")


def _read_api_keys() -> List[str]:
    path = _config_path()
    if not path.exists():
        raise RuntimeError(
            "Revid API keys file not found. "
            "Configure keys via POST /config/revid with {\"api_keys\": [\"key1\", \"key2\"]}."
        )
    raw = path.read_text(encoding="utf-8").strip()
    keys = [k.strip() for k in raw.splitlines() if k.strip()]
    if not keys:
        raise RuntimeError(
            "Revid API keys file is empty. "
            "Configure keys via POST /config/revid."
        )
    return keys


def save_api_keys(api_keys: List[str]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(k.strip() for k in api_keys if k.strip()), encoding="utf-8")
    logger.info("Revid API keys saved (%d keys)", len(api_keys))


def _normalize_segment(seg) -> Optional[dict]:
    """Convert segment to {start, end, text}. Accepts dict or list [start, end, text]."""
    if isinstance(seg, dict):
        start = seg.get("start", "00:00:00,000")
        end = seg.get("end", "00:00:00,000")
        text = (seg.get("text") or "").strip()
    elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
        start, end, text = seg[0], seg[1], (seg[2] or "").strip()
    else:
        return None
    if not text:
        return None
    return {"start": start, "end": end, "text": text}


def _subtitles_to_revid_format(subtitles: list) -> list:
    """Convert project format [{start, end, text}] or [[s,e,t],...] to Revid [{index, start, end, text}]."""
    out = []
    for i, seg in enumerate(subtitles, start=1):
        norm = _normalize_segment(seg)
        if not norm:
            continue
        start_ts = norm["start"]
        end_ts = norm["end"]
        text = norm["text"]
        # Revid expects comma in timestamp (00:00:00,000)
        if "," not in start_ts and "." in start_ts:
            start_ts = start_ts.replace(".", ",", 1)
        if "," not in end_ts and "." in end_ts:
            end_ts = end_ts.replace(".", ",", 1)
        out.append({"index": i, "start": start_ts, "end": end_ts, "text": text})
    return out


def _submit_merge(api_key: str, body: dict) -> str:
    """POST to Revid merge API. Returns task_id. Raises on 402 (caller should try next key)."""
    with httpx.Client(timeout=REVID_REQUEST_TIMEOUT) as client:
        r = client.post(
            REVID_MERGE_URL,
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json=body,
        )
    if r.status_code == 402:
        raise RevidInsufficientCredits("Revid API returned 402: insufficient credits")
    if r.status_code == 401:
        raise RevidUnauthorized("Revid API returned 401: invalid API key")
    if r.status_code != 200:
        msg = r.text[:500] if r.text else "No details"
        raise RuntimeError(f"Revid merge API error (HTTP {r.status_code}): {msg}")
    data = r.json()
    # Some setups may wrap response in a list; unwrap first element if needed
    if isinstance(data, list):
        if not data:
            raise RuntimeError("Revid merge API returned empty list response")
        data = data[0]
    if not isinstance(data, dict):
        raise RuntimeError(f"Revid merge API returned unexpected response type: {type(data).__name__}")

    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError("Revid merge API did not return task_id")
    return task_id


def _get_task(api_key: str, task_id: str) -> dict:
    """GET task status from Revid. Returns full JSON (status, result, etc.)."""
    url = f"{REVID_GET_TASK_URL}/{task_id}"
    with httpx.Client(timeout=REVID_REQUEST_TIMEOUT) as client:
        r = client.get(
            url,
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        )
    if r.status_code != 200:
        msg = r.text[:500] if r.text else "No details"
        raise RuntimeError(f"Revid get task error (HTTP {r.status_code}): {msg}")
    return r.json()


def _download_audio(url: str, path: Path) -> None:
    with httpx.Client(timeout=REVID_REQUEST_TIMEOUT, follow_redirects=True) as client:
        r = client.get(url)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)


class RevidInsufficientCredits(Exception):
    """Raised when Revid returns 402; caller should try next API key."""


class RevidUnauthorized(Exception):
    """Raised when Revid returns 401; API key is invalid."""


class NoActiveKeys(Exception):
    """Raised when no active Revid API keys remain."""


# In-memory key state (safe because JobManager runs jobs sequentially)
_KEY_ORDER: List[str] = []
_KEY_STATE: dict[str, dict] = {}
_KEY_RR_INDEX: int = 0


def _load_keys_and_state() -> None:
    """
    Load keys from file and merge with persisted state.

    For each key we track:
    - status: "active" | "exhausted" | "invalid"
    - last_error: last error string (if any)
    - last_used_at: ISO timestamp string
    - success_count / error_count: simple counters
    """
    global _KEY_ORDER, _KEY_STATE, _KEY_RR_INDEX

    keys = _read_api_keys()
    _KEY_ORDER = keys

    state_path = _state_path()
    if state_path.exists():
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _KEY_STATE = {str(k): dict(v) for k, v in raw.items()}
        except Exception:
            # Corrupted state file – start fresh but don't block processing
            _KEY_STATE = {}

    # Ensure every key has a state entry; drop removed keys
    new_state: dict[str, dict] = {}
    for k in keys:
        entry = _KEY_STATE.get(k, {})
        if "status" not in entry:
            entry["status"] = "active"
        entry.setdefault("last_error", None)
        entry.setdefault("last_used_at", None)
        entry.setdefault("success_count", 0)
        entry.setdefault("error_count", 0)
        new_state[k] = entry
    _KEY_STATE = new_state

    # Reset round-robin index if out of range
    if _KEY_ORDER:
        _KEY_RR_INDEX %= len(_KEY_ORDER)
    else:
        _KEY_RR_INDEX = 0

    _persist_key_state()


def _persist_key_state() -> None:
    """Persist key state to disk."""
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(_KEY_STATE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to persist Revid key state: %s", e)


def _select_next_active_key() -> str:
    """
    Return next active key in round-robin order.

    Skips keys marked as exhausted/invalid. Raises NoActiveKeys if none available.
    """
    global _KEY_RR_INDEX

    if not _KEY_ORDER:
        raise NoActiveKeys("No Revid API keys configured")

    n = len(_KEY_ORDER)
    for _ in range(n):
        idx = _KEY_RR_INDEX
        _KEY_RR_INDEX = (idx + 1) % n
        key = _KEY_ORDER[idx]
        state = _KEY_STATE.get(key, {})
        status = state.get("status", "active")
        if status == "active":
            return key

    raise NoActiveKeys("No active Revid API keys available (all exhausted/invalid)")


def _mark_key(
    key: str,
    status: Optional[str] = None,
    error: Optional[str] = None,
    success: bool = False,
) -> None:
    """Update key state and persist."""
    if key not in _KEY_STATE:
        _KEY_STATE[key] = {
            "status": "active",
            "last_error": None,
            "last_used_at": None,
            "success_count": 0,
            "error_count": 0,
        }
    entry = _KEY_STATE[key]
    if status is not None:
        entry["status"] = status
    if error is not None:
        entry["last_error"] = error
    entry["last_used_at"] = datetime.utcnow().isoformat(timespec="seconds")
    if success:
        entry["success_count"] = int(entry.get("success_count", 0)) + 1
    elif error is not None:
        entry["error_count"] = int(entry.get("error_count", 0)) + 1

    _persist_key_state()


def _process_revid_tts_sync(job: Job, manager: JobManager) -> dict:
    """Blocking: submit to Revid with key rotation, poll, download, return result."""
    params = job.params
    subtitles = params.get("subtitles", [])
    if not subtitles:
        raise ValueError("No subtitles provided.")

    voice_id = params.get("voice_id", 1000)
    speed = float(params.get("speed", 1.0))
    language = params.get("language", "vi-VN")
    add_silence = float(params.get("add_silence", 0.5))

    revid_subtitles = _subtitles_to_revid_format(subtitles)
    if not revid_subtitles:
        raise ValueError("No valid subtitle segments after conversion.")

    body = {
        "subtitles": revid_subtitles,
        "voice_id": voice_id,
        "speed": speed,
        "language": language,
        "add_silence": add_silence,
        "return_url": True,
    }

    # Load keys + state once per job
    _load_keys_and_state()
    task_id = None
    used_key: Optional[str] = None

    # Try at most len(keys) different keys for this job
    attempts = len(_KEY_ORDER)
    if attempts == 0:
        raise RuntimeError("Revid API: no API keys configured in revid_api_keys.txt")

    last_error: Optional[str] = None
    for _ in range(attempts):
        try:
            key = _select_next_active_key()
        except NoActiveKeys as e:
            last_error = str(e)
            break

        try:
            task_id = _submit_merge(key, body)
            used_key = key
            _mark_key(key, status="active", error=None, success=True)
            break
        except RevidInsufficientCredits as e:
            msg = str(e) or "insufficient credits"
            logger.warning("Revid API key exhausted (402), key=***%s, err=%s", key[-4:], msg)
            _mark_key(key, status="exhausted", error=msg, success=False)
            last_error = msg
            continue
        except RevidUnauthorized as e:
            msg = str(e) or "invalid API key"
            logger.warning("Revid API key invalid (401), key=***%s, err=%s", key[-4:], msg)
            _mark_key(key, status="invalid", error=msg, success=False)
            last_error = msg
            continue

    if task_id is None or used_key is None:
        detail = (
            "Revid API: could not create task with any configured API key. "
            "All keys may be exhausted or invalid."
        )
        if last_error:
            detail += f" Last error: {last_error}"
        raise RuntimeError(detail)

    # Poll until completed or failed
    elapsed = 0.0
    while elapsed < REVID_POLL_TIMEOUT:
        result = _get_task(used_key, task_id)
        # Some environments may wrap task object inside a list
        if isinstance(result, list):
            if not result:
                raise RuntimeError("Revid get task returned empty list response")
            result = result[0]
        if not isinstance(result, dict):
            raise RuntimeError(f"Revid get task returned unexpected response type: {type(result).__name__}")

        status = result.get("status", "").lower()

        if status == "completed":
            res = result.get("result")
            # Some responses may wrap result in a list
            if isinstance(res, list):
                res = res[0] if res else {}
            if not isinstance(res, dict):
                res = {}

            audio_url = res.get("audio_url")
            if not audio_url:
                raise RuntimeError("Revid completed but result.audio_url is missing")
            mp3_path = manager.output_dir / f"{job.job_id}.mp3"
            _download_audio(audio_url, mp3_path)
            if not mp3_path.exists():
                raise RuntimeError(f"Failed to save Revid audio to {mp3_path}")
            manager.cleanup_job_work_dir(job)
            return {
                "file_url": f"/static/{job.job_id}.mp3",
                "duration_seconds": res.get("audio_duration"),
                "format": res.get("format", "mp3"),
            }

        if status == "failed":
            msg = result.get("message", "Unknown error")
            raise RuntimeError(f"Revid task failed: {msg}")

        time.sleep(REVID_POLL_INTERVAL)
        elapsed += REVID_POLL_INTERVAL

    raise RuntimeError(f"Revid task {task_id} did not complete within {REVID_POLL_TIMEOUT}s")


async def process_revid_tts(job: Job, manager: JobManager) -> dict:
    """Async wrapper: run blocking Revid flow in executor, progress updates."""
    logger.info("Starting Revid TTS job %s", job.job_id)
    await manager.update_progress(job.job_id, 5)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        _process_revid_tts_sync,
        job,
        manager,
    )
    await manager.update_progress(job.job_id, 100)
    logger.info("Revid TTS job %s completed", job.job_id)
    return result
