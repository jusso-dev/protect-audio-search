"""Tiny stdio MCP server that stands in for mgcrea/mcp-unifi-protect.

Controlled by environment variables so tests stay offline:

- ``MOCK_CALL_LOG``: JSONL file of every export attempt
- ``MOCK_EXPORT_DIR``: where successful exports are written
- ``MOCK_CAMERAS``: JSON list of ``{"id", "name"}`` objects
- ``MOCK_FAIL_FIRST``: fail this many export calls with a transient error
- ``MOCK_PERMANENT_ERROR``: fail every export with this message
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("mock-protect")


def _cameras() -> list[dict[str, str]]:
    raw = os.environ.get("MOCK_CAMERAS")
    if not raw:
        return [{"id": "cam-1", "name": "Street"}, {"id": "cam-2", "name": "Lane"}]
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ToolError("MOCK_CAMERAS was not a list")
    cameras: list[dict[str, str]] = []
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            cameras.append({"id": item["id"], "name": str(item.get("name", ""))})
    return cameras


def _log(record: dict[str, object]) -> int:
    path = Path(os.environ["MOCK_CALL_LOG"])
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    count = sum(1 for line in previous.splitlines() if line.strip())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return count


@mcp.tool(name="unifi_protect_list_cameras")
def list_cameras() -> list[dict[str, str]]:
    """Return the fixture camera list."""
    return _cameras()


@mcp.tool(name="unifi_protect_export_video")
def export_video(cameraId: str, start: str, end: str) -> dict[str, object]:
    """Write a tiny MP4 stand-in and return its path and size."""
    permanent = os.environ.get("MOCK_PERMANENT_ERROR")
    if permanent:
        _log({"cameraId": cameraId, "start": start, "end": end, "error": permanent})
        raise ToolError(permanent)
    fail_first = int(os.environ.get("MOCK_FAIL_FIRST", "0"))
    prior = _log({"cameraId": cameraId, "start": start, "end": end})
    if prior < fail_first:
        raise ToolError("503 temporarily unavailable")
    directory = Path(os.environ["MOCK_EXPORT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    safe = f"{cameraId}-{start.replace(':', '')}-{end.replace(':', '')}.mp4"
    path = directory / safe
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
    return {"path": str(path), "bytes": path.stat().st_size}


if __name__ == "__main__":
    mcp.run()
