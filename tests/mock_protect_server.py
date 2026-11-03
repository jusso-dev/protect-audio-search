"""Tiny stdio MCP server that stands in for mgcrea/mcp-unifi-protect.

Results and errors use the real server's shapes: one JSON text block, no
structured content, and errors as ``{"error": ..., "details": {"status": ...}}``.
Exports honour ``savePath`` and report ``startsAt`` and ``audio``.

Controlled by environment variables so tests stay offline:

- ``MOCK_CALL_LOG``: JSONL file of every export attempt
- ``MOCK_EXPORT_DIR``: where exports go when no ``savePath`` is given
- ``MOCK_CAMERAS``: JSON list of camera objects (``id``, ``name``, ``recordsAudio``)
- ``MOCK_FAIL_FIRST``: fail this many export calls with a transient error
- ``MOCK_PERMANENT_ERROR``: fail every export with this message
- ``MOCK_FAIL_STARTS``: comma-separated ``start`` values that fail permanently
- ``MOCK_MAX_SECONDS``: fail an export longer than this with the download-limit error
- ``MOCK_IGNORE_SAVE_PATH``: write to ``MOCK_EXPORT_DIR`` even when ``savePath`` is given
- ``MOCK_PREROLL_MS``: report ``startsAt`` this many milliseconds before ``start``
- ``MOCK_UNCONFIGURED``: register only ``unifi_protect_auth_status``
- ``MOCK_MEDIA_WAV``: mux this wav into an MP4 instead of a fake file
- ``MOCK_MEDIA_SECONDS``: length of that MP4 (default 3)
- ``MOCK_NO_AUDIO``: with ``MOCK_MEDIA_WAV``, write video only

A JSON file path as the first argument sets the same variables. That is how a
CLI test reaches the mock, because the CLI passes the child an allowlisted
environment that drops ``MOCK_*``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

if len(sys.argv) > 1:
    os.environ.update(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))

mcp = MCPServer("mock-protect")


def _ok(data: object) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(data))])


def _fail(message: str, status: int | None = None) -> CallToolResult:
    payload: dict[str, object] = {"error": message}
    if status is not None:
        payload["details"] = {"status": status}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))], is_error=True
    )


def _cameras() -> list[dict[str, object]]:
    raw = os.environ.get("MOCK_CAMERAS")
    if not raw:
        return [
            {"id": "cam-1", "name": "Street", "recordsAudio": True, "recordingMode": "always"},
            {"id": "cam-2", "name": "Lane", "recordsAudio": True, "recordingMode": "always"},
        ]
    parsed = json.loads(raw)
    return [item for item in parsed if isinstance(item, dict) and isinstance(item.get("id"), str)]


def _log(record: dict[str, object]) -> int:
    path = Path(os.environ["MOCK_CALL_LOG"])
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    count = sum(1 for line in previous.splitlines() if line.strip())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return count


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def list_cameras() -> CallToolResult:
    """Return the fixture camera list."""
    return _ok(_cameras())


def export_video(
    cameraId: str,
    start: str,
    end: str,
    savePath: str | None = None,
    channel: int = 0,
) -> CallToolResult:
    """Write a small MP4 stand-in and return its path and timing."""
    record: dict[str, object] = {
        "cameraId": cameraId,
        "start": start,
        "end": end,
        "savePath": savePath,
        "channel": channel,
    }
    permanent = os.environ.get("MOCK_PERMANENT_ERROR")
    if permanent:
        _log({**record, "error": permanent})
        return _fail(permanent, 404)
    if start in os.environ.get("MOCK_FAIL_STARTS", "").split(","):
        _log({**record, "error": "no recording"})
        return _fail("Protect has no recording for that window", 404)
    max_seconds = os.environ.get("MOCK_MAX_SECONDS")
    span = (_parse(end) - _parse(start)).total_seconds()
    if max_seconds and span > float(max_seconds):
        _log({**record, "error": "limit"})
        return _fail(
            "video/export would return 300000000 bytes, over the 200000000-byte limit. "
            "Narrow the request, or raise UNIFI_PROTECT_MAX_DOWNLOAD_BYTES.",
            None,
        )
    fail_first = int(os.environ.get("MOCK_FAIL_FIRST", "0"))
    prior = _log(record)
    if prior < fail_first:
        return _fail("503 temporarily unavailable", 503)
    if savePath and not os.environ.get("MOCK_IGNORE_SAVE_PATH"):
        path = Path(savePath)
    else:
        directory = Path(os.environ["MOCK_EXPORT_DIR"])
        safe = f"{cameraId}-{start.replace(':', '')}-{end.replace(':', '')}.mp4"
        path = directory / safe
    path.parent.mkdir(parents=True, exist_ok=True)
    has_audio = _write_export(path)
    preroll = timedelta(milliseconds=int(os.environ.get("MOCK_PREROLL_MS", "0")))
    starts_at = _parse(start) - preroll
    return _ok(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "startsAt": starts_at.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "audio": {"durationSeconds": 3.0, "sampleRate": 16000} if has_audio else None,
        }
    )


def auth_status() -> CallToolResult:
    """Report that no console is configured."""
    return _ok({"configured": False})


def _write_export(path: Path) -> bool:
    wav = os.environ.get("MOCK_MEDIA_WAV")
    if not wav:
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
        return True
    seconds = os.environ.get("MOCK_MEDIA_SECONDS", "3")
    no_audio = bool(os.environ.get("MOCK_NO_AUDIO"))
    command = ["ffmpeg", "-y", "-nostdin", "-v", "error"]
    if not no_audio:
        command += ["-stream_loop", "-1", "-i", wav]
    command += ["-f", "lavfi", "-i", f"testsrc=size=32x32:rate=10:duration={seconds}"]
    command += ["-t", seconds]
    if no_audio:
        command += ["-map", "0:v", "-an"]
    else:
        command += ["-map", "1:v", "-map", "0:a", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", str(path)]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"fixture ffmpeg failed: {detail or 'unknown error'}")
    return not no_audio


if os.environ.get("MOCK_UNCONFIGURED"):
    mcp.tool(name="unifi_protect_auth_status")(auth_status)
else:
    mcp.tool(name="unifi_protect_list_cameras")(list_cameras)
    mcp.tool(name="unifi_protect_export_video")(export_video)


if __name__ == "__main__":
    mcp.run()
