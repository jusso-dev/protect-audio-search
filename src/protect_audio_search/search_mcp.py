"""Experimental MCP entry point for an on-demand scan.

Not a supported interface. See docs/search-audio-events.md for the decision
not to proceed beyond this prototype. The process does nothing until a client
calls a tool. Jobs live in memory and die with the process.

Run with ``python -m protect_audio_search.search_mcp``.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from protect_audio_search.audio import prepare_scan
from protect_audio_search.events import EventConfig
from protect_audio_search.footage import redact
from protect_audio_search.output import ScanResult, result_payload
from protect_audio_search.scan import Progress, ScanOutput, plan_chunks, run_scan
from protect_audio_search.times import DEFAULT_TIMEZONE, parse_user_time, zone

ScanRunner = Callable[..., ScanOutput]

mcp = MCPServer("protect-audio-search")


def _real_scan(
    *,
    camera: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    event_config: EventConfig,
    chunk_minutes: int,
    progress: Progress,
) -> ScanOutput:
    prepare_scan()
    return run_scan(
        camera=camera,
        start=start,
        end=end,
        output_dir=output_dir,
        event_config=event_config,
        chunk_minutes=chunk_minutes,
        progress=progress,
    )


@dataclass
class _Job:
    job_id: str
    camera: str
    start: datetime
    end: datetime
    tz_name: str
    output_dir: Path
    chunk_minutes: int
    state: str
    phase: str
    chunks_total: int
    chunk_index: int
    event_count: int | None
    message: str
    result: dict[str, object] | None


class JobStore:
    """One in-memory scan at a time. A finished job does not block the next."""

    def __init__(self, runner: ScanRunner | None = None) -> None:
        self._runner = runner or _real_scan
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start(
        self,
        *,
        camera: str,
        start: str,
        end: str,
        tz_name: str = DEFAULT_TIMEZONE,
        output_dir: str = "exports",
        chunk_minutes: int = 10,
        wait: bool = False,
    ) -> dict[str, object]:
        """Validate, record a job, and start it. `wait` is for tests."""
        name = camera.strip()
        if not name:
            raise ValueError("Camera name or id is required.")
        if chunk_minutes <= 0:
            raise ValueError("chunk_minutes must be a positive number of minutes")
        tz = zone(tz_name)
        start_at = parse_user_time(start, tz)
        end_at = parse_user_time(end, tz)
        if end_at <= start_at:
            raise ValueError("end must be after start")
        planned = plan_chunks(start_at, end_at, chunk_minutes)
        job = _Job(
            job_id=uuid.uuid4().hex[:12],
            camera=name,
            start=start_at,
            end=end_at,
            tz_name=tz_name,
            output_dir=Path(output_dir),
            chunk_minutes=chunk_minutes,
            state="running",
            phase="export",
            chunks_total=len(planned),
            chunk_index=0,
            event_count=None,
            message="",
            result=None,
        )
        with self._lock:
            if any(item.state == "running" for item in self._jobs.values()):
                raise ValueError("A scan is already running.")
            self._jobs[job.job_id] = job
        if wait:
            self._execute(job.job_id)
        else:
            thread = threading.Thread(
                target=self._execute,
                args=(job.job_id,),
                name=f"search-{job.job_id}",
            )
            self._threads[job.job_id] = thread
            thread.start()
        return self.status(job.job_id)

    def status(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Unknown job {job_id!r}.")
            return {
                "job_id": job.job_id,
                "state": job.state,
                "phase": job.phase,
                "chunks_total": job.chunks_total,
                "chunk_index": job.chunk_index,
                "event_count": job.event_count,
                "message": job.message,
            }

    def result(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Unknown job {job_id!r}.")
            state = job.state
            message = job.message
            payload = job.result
        if state == "running":
            raise ValueError("Scan is still running.")
        if state == "failed":
            raise ValueError(message or "Scan failed.")
        if payload is None:
            raise ValueError("Scan result is missing.")
        return {"job_id": job_id, "scan": payload}

    def wait(self, job_id: str, timeout: float = 5) -> None:
        thread = self._threads.get(job_id)
        if thread is None:
            return
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(f"Job {job_id} did not finish.")

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            camera = job.camera
            start = job.start
            end = job.end
            tz_name = job.tz_name
            output_dir = job.output_dir
            chunk_minutes = job.chunk_minutes

        def progress(index: int, count: int, _start: datetime, _end: datetime) -> None:
            with self._lock:
                current = self._jobs[job_id]
                current.phase = "score"
                current.chunks_total = count
                current.chunk_index = index

        try:
            scanned = self._runner(
                camera=camera,
                start=start,
                end=end,
                output_dir=output_dir,
                event_config=EventConfig(),
                chunk_minutes=chunk_minutes,
                progress=progress,
            )
        except Exception as exc:
            message = redact(str(exc)).strip() or "Scan failed."
            with self._lock:
                current = self._jobs[job_id]
                current.state = "failed"
                current.phase = "failed"
                current.message = message
            return
        payload = result_payload(
            ScanResult(
                camera=camera,
                timezone=tz_name,
                start=start,
                end=end,
                events=tuple(scanned.events),
            ),
            zone(tz_name),
        )
        with self._lock:
            current = self._jobs[job_id]
            current.state = "succeeded"
            current.phase = "done"
            current.chunk_index = current.chunks_total
            current.event_count = len(scanned.events)
            current.message = ""
            current.result = payload


STORE = JobStore()


def _tool_call(action: Callable[[], dict[str, object]]) -> dict[str, object]:
    try:
        return action()
    except ValueError as exc:
        raise ToolError(str(exc)) from None


@mcp.tool(name="search_audio_events")
def search_audio_events(
    camera: str,
    start: str,
    end: str,
    tz: str = DEFAULT_TIMEZONE,
    output_dir: str = "exports",
) -> dict[str, object]:
    """Start one on-demand scan. Returns a job and does not wait for the export."""
    return _tool_call(
        lambda: STORE.start(
            camera=camera,
            start=start,
            end=end,
            tz_name=tz,
            output_dir=output_dir,
        )
    )


@mcp.tool(name="search_audio_events_status")
def search_audio_events_status(job_id: str) -> dict[str, object]:
    """Return phase and chunk progress for a scan job."""
    return _tool_call(lambda: STORE.status(job_id))


@mcp.tool(name="search_audio_events_result")
def search_audio_events_result(job_id: str) -> dict[str, object]:
    """Return the scan JSON when the job has finished. Paths only, no image bytes."""
    return _tool_call(lambda: STORE.result(job_id))


def main() -> None:
    """Serve the experimental tools on stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
