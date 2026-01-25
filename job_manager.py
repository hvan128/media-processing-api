"""
Job Manager Module
==================
Manages job lifecycle, state tracking, and sequential execution queue.

Design Decisions:
- Uses asyncio.Queue for sequential job processing (concurrency=1)
- Jobs are stored in-memory dict (sufficient for single-instance deployment)
- Thread-safe access via asyncio locks
- Automatic cleanup of completed job files after configurable TTL
"""

import asyncio
import uuid
import shutil
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional
from dataclasses import dataclass, field


class JobStatus(str, Enum):
    """Job lifecycle states"""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class JobType(str, Enum):
    """Types of processing jobs"""
    STT = "stt"
    SEPARATE = "separate"
    MERGE = "merge"


@dataclass
class Job:
    """
    Represents a processing job with its state and metadata.
    
    Attributes:
        job_id: Unique identifier
        job_type: Type of processing (stt, separate, merge)
        status: Current job status
        progress: Progress percentage (0-100)
        params: Job-specific parameters
        result: Result data when completed
        error_message: Error details if failed
        work_dir: Temporary working directory for this job
        created_at: Job creation timestamp
        completed_at: Job completion timestamp
    """
    job_id: str
    job_type: JobType
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    params: dict = field(default_factory=dict)
    result: Optional[dict] = None
    error_message: Optional[str] = None
    work_dir: Optional[Path] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class JobManager:
    """
    Manages job queue and execution with sequential processing.
    
    Key Features:
    - Single worker ensures sequential execution (concurrency=1)
    - Prevents OOM on memory-constrained VPS
    - Progress tracking for long-running jobs
    - Automatic cleanup of old jobs and temp files
    """
    
    # Base directory for all job data
    DATA_DIR = Path("/data")
    # Fallback to /tmp if /data doesn't exist
    FALLBACK_DIR = Path("/tmp/media_api")
    # Time to keep completed jobs before cleanup
    JOB_TTL_HOURS = 24
    
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._handlers: dict[JobType, Callable] = {}
        self._running = False
        
        # Initialize data directory
        self._data_dir = self.DATA_DIR if self.DATA_DIR.exists() else self.FALLBACK_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        # Output directory for serving files
        self._output_dir = self._data_dir / "output"
        self._output_dir.mkdir(exist_ok=True)
    
    @property
    def output_dir(self) -> Path:
        """Public output directory for completed files"""
        return self._output_dir
    
    def register_handler(
        self, 
        job_type: JobType, 
        handler: Callable[[Job, "JobManager"], Coroutine[Any, Any, dict]]
    ):
        """
        Register a handler function for a job type.
        
        Handler signature: async def handler(job: Job, manager: JobManager) -> dict
        The handler should update job.progress during execution.
        Returns result dict on success, raises exception on failure.
        """
        self._handlers[job_type] = handler
    
    async def create_job(self, job_type: JobType, params: dict) -> Job:
        """
        Create a new job and add it to the processing queue.
        
        Returns immediately with pending job - does NOT wait for processing.
        This enables the async polling pattern required by n8n.
        """
        job_id = str(uuid.uuid4())
        
        # Create job-specific working directory
        work_dir = self._data_dir / "jobs" / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            params=params,
            work_dir=work_dir
        )
        
        async with self._lock:
            self._jobs[job_id] = job
        
        # Add to queue - worker will pick it up
        await self._queue.put(job_id)
        
        return job
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Retrieve job by ID.
        
        This is a pure read operation that MUST return immediately (<10ms).
        Lock is held only for the minimal time needed to access the dict.
        Since asyncio is single-threaded, we can safely return the job reference.
        """
        # Hold lock only for dict lookup - this is extremely fast
        async with self._lock:
            return self._jobs.get(job_id)
    
    async def update_progress(self, job_id: str, progress: int):
        """
        Update job progress (0-100).
        Called by handlers during processing to report progress.
        
        Lock is held only for the minimal time needed to update progress.
        """
        # Clamp progress value
        progress = min(100, max(0, progress))
        
        # Hold lock only for the update operation
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress = progress
    
    async def start(self):
        """Start the job worker and cleanup tasks"""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        asyncio.create_task(self._cleanup_task())
    
    async def stop(self):
        """Gracefully stop the job worker"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    async def _worker(self):
        """
        Sequential job processor.
        
        Processes one job at a time from the queue.
        This is critical for CPU/memory-constrained environments
        where running multiple heavy jobs simultaneously would cause OOM.
        
        IMPORTANT: Lock is held ONLY for brief state updates.
        Handler execution happens WITHOUT lock to ensure GET requests
        never block waiting for job completion.
        """
        while self._running:
            try:
                # Wait for next job (with timeout to allow graceful shutdown)
                try:
                    job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Get job directly from dict (worker is the only writer, safe to read)
                # Hold lock only for the minimal time needed to get job reference
                async with self._lock:
                    job = self._jobs.get(job_id)
                    if not job:
                        self._queue.task_done()
                        continue
                
                # Get handler (no lock needed - handlers are read-only)
                handler = self._handlers.get(job.job_type)
                if not handler:
                    # Update status to error (brief lock)
                    async with self._lock:
                        if job_id in self._jobs:
                            self._jobs[job_id].status = JobStatus.ERROR
                            self._jobs[job_id].error_message = f"No handler for job type: {job.job_type}"
                            self._jobs[job_id].completed_at = datetime.utcnow()
                    self._queue.task_done()
                    continue
                
                # Update status to running (brief lock, then release)
                async with self._lock:
                    if job_id in self._jobs:
                        self._jobs[job_id].status = JobStatus.RUNNING
                        self._jobs[job_id].progress = 0
                
                # CRITICAL: Execute handler WITHOUT lock
                # This ensures GET requests can always read job state
                # even while heavy processing (Demucs, Whisper) is running
                try:
                    result = await handler(job, self)
                    
                    # Mark as done (brief lock)
                    async with self._lock:
                        if job_id in self._jobs:
                            self._jobs[job_id].status = JobStatus.DONE
                            self._jobs[job_id].progress = 100
                            self._jobs[job_id].result = result
                            self._jobs[job_id].completed_at = datetime.utcnow()
                    
                except Exception as e:
                    # Mark as error (brief lock)
                    async with self._lock:
                        if job_id in self._jobs:
                            self._jobs[job_id].status = JobStatus.ERROR
                            self._jobs[job_id].error_message = str(e)
                            self._jobs[job_id].completed_at = datetime.utcnow()
                    
                    # Cleanup work directory on error
                    if job.work_dir and job.work_dir.exists():
                        shutil.rmtree(job.work_dir, ignore_errors=True)
                
                finally:
                    self._queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log but don't crash the worker
                print(f"Worker error: {e}")
    
    async def _cleanup_task(self):
        """
        Periodic cleanup of old jobs and their files.
        
        Runs every hour, removes jobs older than JOB_TTL_HOURS.
        This prevents disk space exhaustion on long-running instances.
        """
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff = datetime.utcnow() - timedelta(hours=self.JOB_TTL_HOURS)
                to_remove = []
                
                async with self._lock:
                    for job_id, job in self._jobs.items():
                        if job.completed_at and job.completed_at < cutoff:
                            to_remove.append(job_id)
                    
                    for job_id in to_remove:
                        job = self._jobs.pop(job_id, None)
                        if job and job.work_dir and job.work_dir.exists():
                            shutil.rmtree(job.work_dir, ignore_errors=True)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup error: {e}")
    
    def cleanup_job_work_dir(self, job: Job):
        """
        Clean up job's working directory (called after output is moved).
        Output files are preserved in the output directory.
        """
        if job.work_dir and job.work_dir.exists():
            shutil.rmtree(job.work_dir, ignore_errors=True)


# Global singleton instance
job_manager = JobManager()
