"""Export UniFi Protect footage in chunks through an MCP server.

The client only calls the two configured read tools. Credentials stay in the
child process environment and are redacted if a tool error echoes them.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

DEFAULT_COMMAND = "npx"
DEFAULT_ARGS: tuple[str, ...] = ("-y", "@mgcrea/mcp-unifi-protect")
DEFAULT_CHUNK_MINUTES = 10
DEFAULT_TOOL_TIMEOUT_SECONDS = 30 * 60

_SECRET_RE = re.compile(r"(?i)((?:password|passwd|token|secret|api[_-]?key)\s*[=:]\s*)(\S+)")
_TRANSIENT_RE = re.compile(
    r"(timeout|timed out|429|500|502|503|504|temporar|connection reset|econnreset)",
    re.IGNORECASE,
)


class ProtectClientError(Exception):
    """Failure talking to the MCP server, or a bad camera or time range."""


class UnknownCameraError(ProtectClientError):
    """The requested camera is not in the server's camera list."""

    def __init__(self, query: str, cameras: Sequence[Camera]) -> None:
        available = ", ".join(f"{camera.name} ({camera.id})" for camera in cameras) or "(none)"
        super().__init__(f"Unknown camera {query!r}. Available cameras: {available}")
        self.query = query
        self.cameras = tuple(cameras)


class ToolFailed(ProtectClientError):
    """The server returned a tool error (`is_error`)."""


@dataclass(frozen=True)
class Camera:
    id: str
    name: str


@dataclass(frozen=True)
class FootageChunk:
    path: Path
    start: datetime
    end: datetime
    skipped: bool


@dataclass(frozen=True)
class ToolNames:
    list_cameras: str = "unifi_protect_list_cameras"
    export_video: str = "unifi_protect_export_video"


def redact(message: str) -> str:
    """Hide credential-shaped values before a message is shown."""
    return _SECRET_RE.sub(r"\1[redacted]", message)


def split_range(
    start: datetime,
    end: datetime,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
) -> list[tuple[datetime, datetime]]:
    """Split `[start, end)` into chunks. The last chunk may be shorter."""
    start_utc = _as_utc(start)
    end_utc = _as_utc(end)
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be a positive number of minutes")
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    step = timedelta(minutes=chunk_minutes)
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start_utc
    while cursor < end_utc:
        nxt = min(cursor + step, end_utc)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


def resolve_camera(cameras: Sequence[Camera], query: str) -> Camera:
    """Match an id exactly, or a name case-insensitively."""
    needle = query.strip()
    if not needle:
        raise UnknownCameraError(query, cameras)
    by_id = [camera for camera in cameras if camera.id == needle]
    if len(by_id) == 1:
        return by_id[0]
    folded = needle.casefold()
    by_name = [camera for camera in cameras if camera.name.casefold() == folded]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise ProtectClientError(
            f"Camera name {query!r} matches {len(by_name)} cameras. Pass a camera id."
        )
    raise UnknownCameraError(query, cameras)


def chunk_destination(output_dir: Path, camera_id: str, start: datetime, end: datetime) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", camera_id).strip("._") or "camera"
    return output_dir / safe_id / f"{_stamp(start)}_{_stamp(end)}.mp4"


def parse_cameras(payload: object) -> list[Camera]:
    data = _unwrap_result(payload)
    if isinstance(data, dict):
        for key in ("cameras", "items"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ProtectClientError("Camera list was not a list")
    cameras: list[Camera] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id", item.get("cameraId"))
        raw_name = item.get("name", item.get("cameraName", ""))
        if isinstance(raw_id, str) and raw_id:
            cameras.append(Camera(id=raw_id, name=str(raw_name)))
    return cameras


def parse_export_path(payload: object) -> Path:
    data = _unwrap_result(payload)
    if not isinstance(data, dict) or "path" not in data:
        raise ProtectClientError("Export result did not include a file path")
    return Path(str(data["path"]))


def is_transient(message: str) -> bool:
    return _TRANSIENT_RE.search(message) is not None


async def export_footage(
    *,
    camera: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    command: str = DEFAULT_COMMAND,
    args: Sequence[str] = DEFAULT_ARGS,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
    tool_names: ToolNames | None = None,
    env: Mapping[str, str] | None = None,
    retries: int = 3,
    retry_backoff_seconds: float = 0.5,
    tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> list[FootageChunk]:
    """Export `camera` over `[start, end)` as ordered chunk files.

    Chunks whose destination file already exists and is non-empty are skipped.
    Transient tool errors are retried with exponential backoff.
    """
    names = tool_names or ToolNames()
    ranges = split_range(start, end, chunk_minutes)
    destination_root = Path(output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ if env is None else env)
    params = StdioServerParameters(command=command, args=list(args), env=child_env)
    try:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=tool_timeout_seconds) as session,
        ):
            try:
                await session.initialize()
            except Exception as exc:
                raise ProtectClientError(
                    redact(f"MCP server failed during startup: {exc}")
                ) from exc
            try:
                listed = await _call_tool(session, names.list_cameras, {})
            except ToolFailed as exc:
                raise ProtectClientError(redact(f"Could not list cameras: {exc}")) from exc
            chosen = resolve_camera(parse_cameras(listed), camera)
            exported: list[FootageChunk] = []
            for chunk_start, chunk_end in ranges:
                dest = chunk_destination(destination_root, chosen.id, chunk_start, chunk_end)
                if dest.is_file() and dest.stat().st_size > 0:
                    exported.append(
                        FootageChunk(
                            path=dest,
                            start=chunk_start,
                            end=chunk_end,
                            skipped=True,
                        )
                    )
                    continue
                source = await _export_with_retry(
                    session,
                    names.export_video,
                    chosen.id,
                    chunk_start,
                    chunk_end,
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                _place_chunk(source, dest)
                exported.append(
                    FootageChunk(path=dest, start=chunk_start, end=chunk_end, skipped=False)
                )
            return exported
    except ProtectClientError:
        raise
    except FileNotFoundError as exc:
        raise ProtectClientError(
            f"Could not start MCP server command {command!r}: {exc}. Check it is on PATH."
        ) from exc
    except Exception as exc:
        # stdio shutdown often wraps the real error in an ExceptionGroup.
        found = _find_client_error(exc)
        if found is not None:
            raise found from None
        raise ProtectClientError(redact(f"MCP server failed to start: {exc}")) from exc


async def list_cameras(
    *,
    command: str = DEFAULT_COMMAND,
    args: Sequence[str] = DEFAULT_ARGS,
    tool_names: ToolNames | None = None,
    env: Mapping[str, str] | None = None,
    tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> list[Camera]:
    """Return cameras from the MCP server. Read-only."""
    names = tool_names or ToolNames()
    child_env = dict(os.environ if env is None else env)
    params = StdioServerParameters(command=command, args=list(args), env=child_env)
    try:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=tool_timeout_seconds) as session,
        ):
            try:
                await session.initialize()
            except Exception as exc:
                raise ProtectClientError(
                    redact(f"MCP server failed during startup: {exc}")
                ) from exc
            try:
                listed = await _call_tool(session, names.list_cameras, {})
            except ToolFailed as exc:
                raise ProtectClientError(redact(f"Could not list cameras: {exc}")) from exc
            return parse_cameras(listed)
    except ProtectClientError:
        raise
    except FileNotFoundError as exc:
        raise ProtectClientError(
            f"Could not start MCP server command {command!r}: {exc}. Check it is on PATH."
        ) from exc
    except Exception as exc:
        found = _find_client_error(exc)
        if found is not None:
            raise found from None
        raise ProtectClientError(redact(f"MCP server failed to start: {exc}")) from exc


def list_cameras_sync(**kwargs: Any) -> list[Camera]:
    """Synchronous wrapper around `list_cameras`."""
    try:
        return asyncio.run(list_cameras(**kwargs))
    except ProtectClientError as exc:
        print(redact(str(exc)), file=sys.stderr)
        raise SystemExit(1) from exc


def export_footage_or_exit(**kwargs: Any) -> list[FootageChunk]:
    """Run `export_footage` and exit 1 with a readable message on failure."""
    try:
        return asyncio.run(export_footage(**kwargs))
    except ProtectClientError as exc:
        print(redact(str(exc)), file=sys.stderr)
        raise SystemExit(1) from exc


async def _export_with_retry(
    session: ClientSession,
    tool_name: str,
    camera_id: str,
    start: datetime,
    end: datetime,
    *,
    retries: int,
    retry_backoff_seconds: float,
) -> Path:
    attempts = retries + 1
    last_error: ToolFailed | None = None
    for attempt in range(attempts):
        try:
            payload = await _call_tool(
                session,
                tool_name,
                {
                    "cameraId": camera_id,
                    "start": _iso(start),
                    "end": _iso(end),
                },
            )
            source = parse_export_path(payload)
            if not source.is_file() or source.stat().st_size <= 0:
                raise ProtectClientError(f"Export tool reported {source} but the file is missing")
            return source
        except ToolFailed as exc:
            last_error = exc
            if not is_transient(str(exc)) or attempt >= retries:
                raise ProtectClientError(redact(f"Export failed: {exc}")) from exc
            await asyncio.sleep(retry_backoff_seconds * (2**attempt))
    raise ProtectClientError(redact(f"Export failed: {last_error}"))


async def _call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> object:
    try:
        result = await session.call_tool(name, arguments)
    except Exception as exc:
        raise ProtectClientError(redact(f"Tool {name} failed: {exc}")) from exc
    if not isinstance(result, CallToolResult):
        raise ProtectClientError(f"Tool {name} returned an unexpected result")
    if result.is_error:
        raise ToolFailed(_result_text(result) or f"Tool {name} failed")
    if result.structured_content is not None:
        return _unwrap_result(result.structured_content)
    text = _result_text(result)
    try:
        return _unwrap_result(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ProtectClientError(
            f"Tool {name} returned non-JSON text: {redact(text)[:500]}"
        ) from exc


def _place_chunk(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == dest.resolve():
        return
    partial = dest.with_suffix(".mp4.partial")
    shutil.copyfile(source, partial)
    os.replace(partial, dest)


def _unwrap_result(payload: object) -> object:
    if isinstance(payload, dict) and list(payload) == ["result"]:
        return cast(object, payload["result"])
    return payload


def _result_text(result: CallToolResult) -> str:
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _find_client_error(
    exc: BaseException,
    seen: set[int] | None = None,
) -> ProtectClientError | None:
    """Return a client error nested inside a transport ExceptionGroup, if any."""
    seen = seen or set()
    if id(exc) in seen:
        return None
    seen.add(id(exc))
    if isinstance(exc, ProtectClientError):
        return exc
    nested: list[BaseException] = []
    if isinstance(exc, BaseExceptionGroup):
        nested.extend(exc.exceptions)
    if exc.__cause__ is not None:
        nested.append(exc.__cause__)
    if exc.__context__ is not None:
        nested.append(exc.__context__)
    for item in nested:
        found = _find_client_error(item, seen)
        if found is not None:
            return found
    return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetimes must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _as_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(value: datetime) -> str:
    return _as_utc(value).strftime("%Y%m%dT%H%M%SZ")
