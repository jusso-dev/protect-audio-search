"""In-memory scan jobs. No Protect console and no model download."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from protect_audio_search.events import EventConfig
from protect_audio_search.scan import Progress, ScanOutput
from protect_audio_search.search_mcp import JobStore, mcp, search_audio_events

START = "2026-10-20 21:00"
END = "2026-10-20 21:20"


def _quiet(
    *,
    camera: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    event_config: EventConfig,
    chunk_minutes: int,
    progress: Progress,
) -> ScanOutput:
    assert camera == "Street"
    assert output_dir == Path("exports")
    assert chunk_minutes == 10
    assert event_config.threshold == 0.3
    progress(1, 2, start, end)
    progress(2, 2, start, end)
    return ScanOutput(events=[], chunks=[])


def test_finished_job_matches_scan_schema() -> None:
    store = JobStore(runner=_quiet)
    status = store.start(camera=" Street ", start=START, end=END, wait=True)
    assert status["state"] == "succeeded"
    assert status["phase"] == "done"
    assert status["chunks_total"] == 2
    assert status["chunk_index"] == 2
    assert status["event_count"] == 0
    body = store.result(str(status["job_id"]))
    scan = body["scan"]
    assert isinstance(scan, dict)
    assert scan["schema_version"] == 1
    assert scan["camera"] == "Street"
    assert scan["timezone"] == "Australia/Sydney"
    assert scan["start_utc"] == "2026-10-20T10:00:00Z"
    assert scan["end_utc"] == "2026-10-20T10:20:00Z"
    assert scan["events"] == []


def test_export_phase_blocks_a_second_scan() -> None:
    started = threading.Event()
    release = threading.Event()

    def runner(
        *,
        camera: str,
        start: datetime,
        end: datetime,
        output_dir: Path,
        event_config: EventConfig,
        chunk_minutes: int,
        progress: Progress,
    ) -> ScanOutput:
        del camera, start, end, output_dir, event_config, chunk_minutes, progress
        started.set()
        assert release.wait(timeout=5)
        return ScanOutput(events=[], chunks=[])

    store = JobStore(runner=runner)
    status = store.start(camera="Street", start=START, end=END)
    assert started.wait(timeout=5)
    try:
        assert status["state"] == "running"
        assert status["phase"] == "export"
        assert status["chunk_index"] == 0
        assert status["chunks_total"] == 2
        with pytest.raises(ValueError, match="already running"):
            store.start(camera="Street", start=START, end=END)
        with pytest.raises(ValueError, match="still running"):
            store.result(str(status["job_id"]))
    finally:
        release.set()
        store.wait(str(status["job_id"]))
    done = store.status(str(status["job_id"]))
    assert done["state"] == "succeeded"


def test_failure_is_redacted_and_does_not_block_the_next_job() -> None:
    calls = {"count": 0}

    def runner(
        *,
        camera: str,
        start: datetime,
        end: datetime,
        output_dir: Path,
        event_config: EventConfig,
        chunk_minutes: int,
        progress: Progress,
    ) -> ScanOutput:
        del camera, start, end, output_dir, event_config, chunk_minutes, progress
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("password=hunter2")
        return ScanOutput(events=[], chunks=[])

    store = JobStore(runner=runner)
    failed = store.start(camera="Street", start=START, end="2026-10-20 21:10", wait=True)
    assert failed["state"] == "failed"
    assert failed["phase"] == "failed"
    assert "hunter2" not in str(failed["message"])
    assert "password=[redacted]" in str(failed["message"])
    with pytest.raises(ValueError, match="redacted"):
        store.result(str(failed["job_id"]))
    again = store.start(camera="Street", start=START, end="2026-10-20 21:10", wait=True)
    assert again["state"] == "succeeded"


def test_bad_times_do_not_start_a_job() -> None:
    store = JobStore(runner=_quiet)
    with pytest.raises(ValueError, match="end must be after start"):
        store.start(camera="Street", start=END, end=START, wait=True)
    with pytest.raises(ValueError, match="Could not parse"):
        store.start(camera="Street", start="tomorrow", end=END, wait=True)
    with pytest.raises(ValueError, match="Camera name"):
        store.start(camera="  ", start=START, end=END, wait=True)
    with pytest.raises(ValueError, match="Unknown job"):
        store.status("missing")


def test_tool_turns_validation_errors_into_tool_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("protect_audio_search.search_mcp.STORE", JobStore(runner=_quiet))
    with pytest.raises(ToolError, match="end must be after start"):
        search_audio_events(camera="Street", start=END, end=START)


def test_registered_tool_names() -> None:
    tools = asyncio.run(mcp.list_tools())
    assert sorted(tool.name for tool in tools) == [
        "search_audio_events",
        "search_audio_events_result",
        "search_audio_events_status",
    ]
