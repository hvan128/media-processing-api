"""
Eco88Labs TTS Service
=====================
Text-to-Speech via Eco88Labs API (https://eco88labs.com/api/v1).

- Submits gen_text to /tts-infer (async), polls /tts/status/<task_id> until completed,
  downloads audio to output_dir.
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

ECO88_BASE_URL = "https://eco88labs.com/api/v1"
ECO88_INFER_URL = f"{ECO88_BASE_URL}/tts-infer"
ECO88_STATUS_URL = f"{ECO88_BASE_URL}/tts/status"
ECO88_POLL_INTERVAL = 3.0
ECO88_POLL_TIMEOUT = 300.0  # 5 minutes
ECO88_REQUEST_TIMEOUT = 60.0

CONFIG_DIR = Path("/data/config")
ECO88_KEYS_FILE = CONFIG_DIR / "eco88labs_api_keys.txt"
FALLBACK_CONFIG = Path(__file__).resolve().parent / "data" / "config" / "eco88labs_api_keys.txt"


def _config_path() -> Path:
    if ECO88_KEYS_FILE.exists():
        return ECO88_KEYS_FILE
    return FALLBACK_CONFIG


def _state_path() -> Path:
    """Path to persisted key state JSON (same directory as keys file)."""
    return _config_path().with_name("eco88labs_key_state.json")


def _read_api_keys() -> List[str]:
    path = _config_path()
    if not path.exists():
        raise RuntimeError(
            "Eco88Labs API keys file not found. "
            "Configure keys via POST /config/eco88labs with {\"api_keys\": [\"key1\", \"key2\"]}."
        )
    raw = path.read_text(encoding="utf-8").strip()
    keys = [k.strip() for k in raw.splitlines() if k.strip()]
    if not keys:
        raise RuntimeError(
            "Eco88Labs API keys file is empty. "
            "Configure keys via POST /config/eco88labs."
        )
    return keys


def get_any_active_key() -> str:
    """Load keys/state and return first active key. Used for read-only API calls (e.g. voices)."""
    _load_keys_and_state()
    return _select_next_active_key()


def get_voices(api_key: str) -> list:
    """Fetch available voices from Eco88Labs API."""
    url = f"{ECO88_BASE_URL}/tts/voices"
    with httpx.Client(timeout=ECO88_REQUEST_TIMEOUT) as client:
        r = client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    if r.status_code == 401:
        raise Eco88Unauthorized("Invalid API key")
    r.raise_for_status()
    data = r.json()
    return data.get("voices", data)


def save_api_keys(api_keys: List[str]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(k.strip() for k in api_keys if k.strip()), encoding="utf-8")
    logger.info("Eco88Labs API keys saved (%d keys)", len(api_keys))


class Eco88InsufficientBalance(Exception):
    """Raised when Eco88Labs returns 402 (INSUFFICIENT_TOKEN_BALANCE); caller should try next key."""


class Eco88Unauthorized(Exception):
    """Raised when Eco88Labs returns 401; API key is invalid."""


class NoActiveKeys(Exception):
    """Raised when no active Eco88Labs API keys remain."""


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
            _KEY_STATE = {}

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
        logger.warning("Failed to persist Eco88Labs key state: %s", e)


def _select_next_active_key() -> str:
    """
    Return next active key in round-robin order.

    Skips keys marked as exhausted/invalid. Raises NoActiveKeys if none available.
    """
    global _KEY_RR_INDEX

    if not _KEY_ORDER:
        raise NoActiveKeys("No Eco88Labs API keys configured")

    n = len(_KEY_ORDER)
    for _ in range(n):
        idx = _KEY_RR_INDEX
        _KEY_RR_INDEX = (idx + 1) % n
        key = _KEY_ORDER[idx]
        state = _KEY_STATE.get(key, {})
        status = state.get("status", "active")
        if status == "active":
            return key

    raise NoActiveKeys("No active Eco88Labs API keys available (all exhausted/invalid)")


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


def _submit_infer(api_key: str, body: dict) -> str:
    """POST to /tts-infer. Returns task_id. Raises on 402/401."""
    with httpx.Client(timeout=ECO88_REQUEST_TIMEOUT) as client:
        r = client.post(
            ECO88_INFER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
    if r.status_code == 402:
        raise Eco88InsufficientBalance("Eco88Labs API returned 402: insufficient token balance")
    if r.status_code == 401:
        raise Eco88Unauthorized("Eco88Labs API returned 401: invalid API key")
    if r.status_code != 202:
        msg = r.text[:500] if r.text else "No details"
        raise RuntimeError(f"Eco88Labs tts-infer error (HTTP {r.status_code}): {msg}")
    data = r.json()
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError("Eco88Labs tts-infer did not return task_id")
    return task_id


def _get_status(api_key: str, task_id: str) -> dict:
    """GET /tts/status/<task_id>. Returns full JSON."""
    url = f"{ECO88_STATUS_URL}/{task_id}"
    with httpx.Client(timeout=ECO88_REQUEST_TIMEOUT) as client:
        r = client.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if r.status_code != 200:
        msg = r.text[:500] if r.text else "No details"
        raise RuntimeError(f"Eco88Labs status check error (HTTP {r.status_code}): {msg}")
    return r.json()


def _download_audio(url: str, path: Path) -> None:
    with httpx.Client(timeout=ECO88_REQUEST_TIMEOUT, follow_redirects=True) as client:
        r = client.get(url)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)


def _process_eco88labs_tts_sync(job: Job, manager: JobManager) -> dict:
    """Blocking: submit to Eco88Labs with key rotation, poll, download, return result."""
    params = job.params
    gen_text = params.get("gen_text", "").strip()
    if not gen_text:
        raise ValueError("gen_text is required and cannot be empty.")

    name_character = params.get("name_character", "")
    if not name_character:
        raise ValueError("name_character is required.")

    body = {
        "gen_text": gen_text,
        "name_character": name_character,
    }
    output_format = params.get("output_format")
    if output_format:
        body["output_format"] = output_format
    sample_rate = params.get("sample_rate")
    if sample_rate is not None:
        body["sample_rate"] = sample_rate
    speed = params.get("speed")
    if speed is not None:
        body["speed"] = str(speed)
    seed = params.get("seed")
    if seed is not None:
        body["seed"] = seed

    # Load keys + state once per job
    _load_keys_and_state()
    task_id = None
    used_key: Optional[str] = None

    attempts = len(_KEY_ORDER)
    if attempts == 0:
        raise RuntimeError("Eco88Labs TTS: no API keys configured in eco88labs_api_keys.txt")

    last_error: Optional[str] = None
    for _ in range(attempts):
        try:
            key = _select_next_active_key()
        except NoActiveKeys as e:
            last_error = str(e)
            break

        try:
            task_id = _submit_infer(key, body)
            used_key = key
            _mark_key(key, status="active", error=None, success=True)
            break
        except Eco88InsufficientBalance as e:
            msg = str(e) or "insufficient token balance"
            logger.warning("Eco88Labs key exhausted (402), key=***%s, err=%s", key[-4:], msg)
            _mark_key(key, status="exhausted", error=msg, success=False)
            last_error = msg
            continue
        except Eco88Unauthorized as e:
            msg = str(e) or "invalid API key"
            logger.warning("Eco88Labs key invalid (401), key=***%s, err=%s", key[-4:], msg)
            _mark_key(key, status="invalid", error=msg, success=False)
            last_error = msg
            continue

    if task_id is None or used_key is None:
        detail = (
            "Eco88Labs TTS: could not create task with any configured API key. "
            "All keys may be exhausted or invalid."
        )
        if last_error:
            detail += f" Last error: {last_error}"
        raise RuntimeError(detail)

    # Poll until completed or failed
    fmt = (output_format or "wav").lstrip(".")
    elapsed = 0.0
    while elapsed < ECO88_POLL_TIMEOUT:
        data = _get_status(used_key, task_id)
        status = data.get("status", "").lower()

        if status == "completed":
            audio_url = data.get("output_file_url")
            if not audio_url:
                raise RuntimeError("Eco88Labs task completed but output_file_url is missing")
            out_path = manager.output_dir / f"{job.job_id}.{fmt}"
            _download_audio(audio_url, out_path)
            if not out_path.exists():
                raise RuntimeError(f"Failed to save Eco88Labs audio to {out_path}")
            manager.cleanup_job_work_dir(job)
            return {
                "file_url": f"/static/{job.job_id}.{fmt}",
                "format": fmt,
            }

        if status == "failed":
            msg = data.get("error") or "Unknown error"
            raise RuntimeError(f"Eco88Labs task failed: {msg}")

        time.sleep(ECO88_POLL_INTERVAL)
        elapsed += ECO88_POLL_INTERVAL

    raise RuntimeError(f"Eco88Labs task {task_id} did not complete within {ECO88_POLL_TIMEOUT}s")


async def process_eco88labs_tts(job: Job, manager: JobManager) -> dict:
    """Async wrapper: run blocking Eco88Labs flow in executor."""
    logger.info("Starting Eco88Labs TTS job %s", job.job_id)
    await manager.update_progress(job.job_id, 5)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        _process_eco88labs_tts_sync,
        job,
        manager,
    )
    await manager.update_progress(job.job_id, 100)
    logger.info("Eco88Labs TTS job %s completed", job.job_id)
    return result
