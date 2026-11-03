"""Export UniFi Protect footage in chunks through an MCP server.

The client only calls the two configured read tools. Credentials stay in the
child process environment and are redacted if a tool error echoes them. The
child gets an allowlisted copy of the environment, not all of it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult

# Pinned so a cold npx cache cannot pull an unreviewed release.
SERVER_PACKAGE = "@mgcrea/mcp-unifi-protect@1.1.0"
DEFAULT_COMMAND = "npx"
DEFAULT_ARGS: tuple[str, ...] = ("-y", SERVER_PACKAGE)
COMMAND_ENV = "PROTECT_AUDIO_SEARCH_MCP_COMMAND"
DEFAULT_CHUNK_MINUTES = 10
DEFAULT_CHANNEL = 0
DEFAULT_TOOL_TIMEOUT_SECONDS = 15 * 60
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
# A chunk over the server's download cap is split in half, down to this length.
MIN_SPLIT_SECONDS = 60
# Give up on a run when this many chunks in a row fail to export.
MAX_CONSECUTIVE_FAILURES = 3

_SECRET_KEYS = r"password|passwd|passcode|token|secret|api[_-]?key|totp|authorization|cookie"
_SECRET_RE = re.compile(
    rf"(?i)([\"']?\b(?:{_SECRET_KEYS})\b[\"']?\s*[=:]\s*)"
    r"((?:(?:bearer|basic)\s+)?(?:\"(?:[^\"\\]|\\.)*\"|'[^']*'|[^\s,;}]+))"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")
_URL_AUTH_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@")
# Literal values of these variables are scrubbed from any message we print.
_SECRET_ENV = (
    "UNIFI_PROTECT_PASSWORD",
    "UNIFI_PROTECT_API_KEY",
    "UNIFI_PROTECT_TOTP",
    "UNIFI_PROTECT_TOTP_SECRET",
    "ANTHROPIC_API_KEY",
)
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_TRANSIENT_RE = re.compile(
    r"(\btime(?:d)?[ -]?out\b|\btemporar|\bconnection reset\b|\beconnreset\b|\betimedout\b"
    r"|\beconnrefused\b|\bfetch failed\b|\bsocket hang up\b|\b(?:429|50[0234])\b)",
    re.IGNORECASE,
)
# JSON-RPC code the MCP SDK uses when a request outlives its read timeout.
_REQUEST_TIMEOUT = -32001
_DOWNLOAD_LIMIT_RE = re.compile(r"-byte limit|UNIFI_PROTECT_MAX_DOWNLOAD_BYTES", re.IGNORECASE)

# Variables the MCP server child may see. Everything else stays in this process.
_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "LANG",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "APPDATA",
        "LOCALAPPDATA",
        "USERPROFILE",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "HOMEDRIVE",
        "HOMEPATH",
        "PATHEXT",
        "COMSPEC",
    }
)
_ENV_PREFIXES = ("UNIFI_PROTECT_", "LC_", "NODE_", "NPM_CONFIG_", "npm_config_", "XDG_", "NVM_")

ExportProgress = Callable[[int, int, datetime, datetime, bool], None]
ChunkReady = Callable[["FootageChunk"], None]
ChunkFailed = Callable[["FailedChunk"], None]


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
    """The server returned a tool error (`is_error`), or the call timed out."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        transient: bool | None = None,
        needs_two_factor: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.needs_two_factor = needs_two_factor
        if transient is None:
            transient = (status in _TRANSIENT_STATUS) if status is not None else None
        self.transient = is_transient(message) if transient is None else transient


@dataclass(frozen=True)
class Camera:
    id: str
    name: str
    records_audio: bool | None = None
    recording_mode: str | None = None
    is_connected: bool | None = None


@dataclass(frozen=True)
class FootageChunk:
    """One exported file. `media_start` is when its first frame was recorded."""

    path: Path
    start: datetime
    end: datetime
    skipped: bool
    media_start: datetime | None = None
    has_audio: bool | None = None

    @property
    def origin(self) -> datetime:
        """Wall-clock time of offset 0 in the file."""
        return self.media_start or self.start


@dataclass(frozen=True)
class FailedChunk:
    start: datetime
    end: datetime
    error: str


@dataclass(frozen=True)
class ExportInfo:
    path: Path
    media_start: datetime | None
    has_audio: bool | None


@dataclass(frozen=True)
class ToolNames:
    list_cameras: str = "unifi_protect_list_cameras"
    export_video: str = "unifi_protect_export_video"


def redact(message: str) -> str:
    """Hide credential-shaped values before a message is shown."""
    text = _SECRET_RE.sub(_mask_secret, message)
    text = _BEARER_RE.sub(r"\1[redacted]", text)
    text = _URL_AUTH_RE.sub(r"\1[redacted]@", text)
    for name in _SECRET_ENV:
        value = os.environ.get(name, "")
        if len(value) >= 4:
            text = text.replace(value, "[redacted]")
    return text


def _mask_secret(match: re.Match[str]) -> str:
    value = match.group(2)
    quote = value[0] if value[:1] in {'"', "'"} else ""
    return f"{match.group(1)}{quote}[redacted]{quote}"


def server_command(environ: Mapping[str, str] | None = None) -> tuple[str, tuple[str, ...]]:
    """The MCP server launch command. `PROTECT_AUDIO_SEARCH_MCP_COMMAND` overrides it."""
    source = os.environ if environ is None else environ
    raw = source.get(COMMAND_ENV, "").strip()
    if not raw:
        return DEFAULT_COMMAND, DEFAULT_ARGS
    parts = shlex.split(raw)
    return parts[0], tuple(parts[1:])


def child_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy only the variables the MCP server needs. API keys for other services stay here."""
    source = os.environ if environ is None else environ
    return {
        key: value
        for key, value in source.items()
        if key in _ENV_NAMES or key.startswith(_ENV_PREFIXES)
    }


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


def check_records_audio(camera: Camera) -> None:
    """Refuse a camera the console says keeps no sound. Unknown passes."""
    if camera.records_audio is False:
        raise ProtectClientError(
            f"Camera {camera.name!r} does not record audio (no microphone, microphone off, "
            "audio not kept in recordings, or recording mode never). Nothing was exported. "
            "Turn on the microphone and audio recording in Protect for future footage."
        )


def chunk_destination(output_dir: Path, camera_id: str, start: datetime, end: datetime) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", camera_id).strip("._") or "camera"
    return output_dir / safe_id / f"{_stamp(start)}_{_stamp(end)}.mp4"


def sidecar_path(chunk_path: Path) -> Path:
    """Export metadata kept next to a chunk so a resumed run keeps `startsAt`."""
    return chunk_path.with_suffix(".json")


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
            mode = item.get("recordingMode")
            cameras.append(
                Camera(
                    id=raw_id,
                    name=str(raw_name),
                    records_audio=_optional_bool(item.get("recordsAudio")),
                    recording_mode=mode if isinstance(mode, str) else None,
                    is_connected=_optional_bool(item.get("isConnected")),
                )
            )
    return cameras


def parse_export_path(payload: object) -> Path:
    return parse_export(payload).path


def parse_export(payload: object) -> ExportInfo:
    data = _unwrap_result(payload)
    if not isinstance(data, dict) or "path" not in data:
        raise ProtectClientError("Export result did not include a file path")
    media_start: datetime | None = None
    starts_at = data.get("startsAt")
    if isinstance(starts_at, str):
        try:
            media_start = _parse_iso(starts_at)
        except ValueError:
            media_start = None
    has_audio: bool | None = None
    if "audio" in data:
        has_audio = data["audio"] is not None
    return ExportInfo(path=Path(str(data["path"])), media_start=media_start, has_audio=has_audio)


def is_transient(message: str) -> bool:
    return _TRANSIENT_RE.search(message) is not None


def is_download_limit(message: str) -> bool:
    return _DOWNLOAD_LIMIT_RE.search(message) is not None


async def export_footage(
    *,
    camera: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    command: str | None = None,
    args: Sequence[str] | None = None,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
    channel: int = DEFAULT_CHANNEL,
    tool_names: ToolNames | None = None,
    env: Mapping[str, str] | None = None,
    retries: int = 3,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    require_audio: bool = False,
    progress: ExportProgress | None = None,
    on_chunk: ChunkReady | None = None,
    on_failure: ChunkFailed | None = None,
) -> list[FootageChunk]:
    """Export `camera` over `[start, end)` as ordered chunk files.

    Chunks whose destination file already exists and is non-empty are skipped.
    Transient tool errors are retried with exponential backoff. A chunk over the
    server's download cap is split in half and retried. `on_chunk` runs as each
    chunk lands, so a caller can score it before the next export.

    With `on_failure`, a chunk that still fails is reported and the run carries
    on, until `MAX_CONSECUTIVE_FAILURES` chunks in a row fail. Without it, the
    first failure raises.
    """
    names = tool_names or ToolNames()
    ranges = split_range(start, end, chunk_minutes)
    destination_root = Path(output_dir).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    launch, launch_args = _launch(command, args)
    child_env = child_environment() if env is None else dict(env)
    params = StdioServerParameters(command=launch, args=list(launch_args), env=child_env)
    try:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=tool_timeout_seconds) as session,
        ):
            await _start_session(session, names)
            chosen = resolve_camera(await _list(session, names), camera)
            if require_audio:
                check_records_audio(chosen)
            exported: list[FootageChunk] = []
            consecutive = 0
            for index, (chunk_start, chunk_end) in enumerate(ranges, start=1):
                pending = [(chunk_start, chunk_end)]
                while pending:
                    part_start, part_end = pending.pop(0)
                    dest = chunk_destination(destination_root, chosen.id, part_start, part_end)
                    cached = _cached_chunk(dest, part_start, part_end)
                    if progress is not None:
                        progress(index, len(ranges), part_start, part_end, cached is not None)
                    if cached is not None:
                        exported.append(cached)
                        consecutive = 0
                        if on_chunk is not None:
                            on_chunk(cached)
                        continue
                    try:
                        info = await _export_with_retry(
                            session,
                            names.export_video,
                            chosen.id,
                            part_start,
                            part_end,
                            dest=dest,
                            channel=channel,
                            retries=retries,
                            retry_backoff_seconds=retry_backoff_seconds,
                        )
                    except ProtectClientError as exc:
                        halves = _split_for_limit(str(exc), part_start, part_end)
                        if halves:
                            pending[:0] = halves
                            continue
                        if on_failure is None:
                            raise
                        consecutive += 1
                        on_failure(FailedChunk(part_start, part_end, redact(str(exc))))
                        if consecutive >= MAX_CONSECUTIVE_FAILURES:
                            raise ProtectClientError(
                                f"{consecutive} chunks in a row failed to export. "
                                f"Last error: {redact(str(exc))}"
                            ) from exc
                        continue
                    chunk = _place_chunk(info, dest, part_start, part_end)
                    exported.append(chunk)
                    consecutive = 0
                    if on_chunk is not None:
                        on_chunk(chunk)
            return exported
    except ProtectClientError:
        raise
    except FileNotFoundError as exc:
        raise ProtectClientError(
            f"Could not start MCP server command {launch!r}: {exc}. Check it is on PATH."
        ) from exc
    except Exception as exc:
        # stdio shutdown often wraps the real error in an ExceptionGroup.
        found = _find_client_error(exc)
        if found is not None:
            raise found from None
        raise ProtectClientError(redact(f"MCP server failed to start: {exc}")) from exc


async def list_cameras(
    *,
    command: str | None = None,
    args: Sequence[str] | None = None,
    tool_names: ToolNames | None = None,
    env: Mapping[str, str] | None = None,
    tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> list[Camera]:
    """Return cameras from the MCP server. Read-only."""
    names = tool_names or ToolNames()
    launch, launch_args = _launch(command, args)
    child_env = child_environment() if env is None else dict(env)
    params = StdioServerParameters(command=launch, args=list(launch_args), env=child_env)
    try:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=tool_timeout_seconds) as session,
        ):
            await _start_session(session, names)
            return await _list(session, names)
    except ProtectClientError:
        raise
    except FileNotFoundError as exc:
        raise ProtectClientError(
            f"Could not start MCP server command {launch!r}: {exc}. Check it is on PATH."
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


def _launch(command: str | None, args: Sequence[str] | None) -> tuple[str, tuple[str, ...]]:
    if command is not None:
        return command, tuple(DEFAULT_ARGS if args is None else args)
    default_command, default_args = server_command()
    return default_command, tuple(default_args if args is None else args)


async def _start_session(session: ClientSession, names: ToolNames) -> None:
    try:
        await session.initialize()
    except Exception as exc:
        raise ProtectClientError(redact(f"MCP server failed during startup: {exc}")) from exc
    try:
        listed = await session.list_tools()
    except Exception as exc:
        raise ProtectClientError(redact(f"MCP server could not list its tools: {exc}")) from exc
    offered = {tool.name for tool in listed.tools}
    missing = [name for name in (names.list_cameras, names.export_video) if name not in offered]
    if missing:
        raise ProtectClientError(
            f"The MCP server does not offer {', '.join(missing)}. This usually means no "
            "console is configured. Set UNIFI_PROTECT_HOST, UNIFI_PROTECT_USERNAME and "
            "UNIFI_PROTECT_PASSWORD (or the cloud variables) in this shell. See the README."
        )


async def _list(session: ClientSession, names: ToolNames) -> list[Camera]:
    try:
        listed = await _call_tool(session, names.list_cameras, {})
    except ToolFailed as exc:
        raise ProtectClientError(redact(f"Could not list cameras: {_explain(exc)}")) from exc
    return parse_cameras(listed)


def _cached_chunk(dest: Path, start: datetime, end: datetime) -> FootageChunk | None:
    if not dest.is_file() or dest.stat().st_size <= 0:
        return None
    media_start: datetime | None = None
    has_audio: bool | None = None
    try:
        meta = json.loads(sidecar_path(dest).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    if isinstance(meta, dict):
        raw_start = meta.get("media_start")
        if isinstance(raw_start, str):
            try:
                media_start = _parse_iso(raw_start)
            except ValueError:
                media_start = None
        has_audio = _optional_bool(meta.get("has_audio"))
    return FootageChunk(
        path=dest,
        start=start,
        end=end,
        skipped=True,
        media_start=media_start,
        has_audio=has_audio,
    )


def _split_for_limit(
    message: str, start: datetime, end: datetime
) -> list[tuple[datetime, datetime]]:
    """Halves of an over-limit chunk, or nothing if it is not over the limit or too short."""
    if not is_download_limit(message):
        return []
    span = (end - start).total_seconds()
    if span < 2 * MIN_SPLIT_SECONDS:
        return []
    middle = start + timedelta(seconds=int(span // 2))
    return [(start, middle), (middle, end)]


async def _export_with_retry(
    session: ClientSession,
    tool_name: str,
    camera_id: str,
    start: datetime,
    end: datetime,
    *,
    dest: Path,
    channel: int,
    retries: int,
    retry_backoff_seconds: float,
) -> ExportInfo:
    partial = _partial_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempts = retries + 1
    last_error: ToolFailed | None = None
    try:
        for attempt in range(attempts):
            try:
                payload = await _call_tool(
                    session,
                    tool_name,
                    {
                        "cameraId": camera_id,
                        "start": _iso(start),
                        "end": _iso(end),
                        "savePath": str(partial),
                        "channel": channel,
                    },
                )
                info = parse_export(payload)
                if not _same_file(info.path, partial):
                    raise ProtectClientError(
                        f"Export tool wrote {info.path}, not the requested {partial}. "
                        "That file was left where the server put it."
                    )
                if not partial.is_file() or partial.stat().st_size <= 0:
                    raise ProtectClientError(
                        f"Export tool reported {info.path} but the file is missing"
                    )
                return info
            except ToolFailed as exc:
                last_error = exc
                if not exc.transient or attempt >= retries:
                    raise ProtectClientError(redact(f"Export failed: {_explain(exc)}")) from exc
                await asyncio.sleep(retry_backoff_seconds * (2**attempt))
        raise ProtectClientError(redact(f"Export failed: {last_error}"))
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


async def _call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> object:
    try:
        result = await session.call_tool(name, arguments)
    except MCPError as exc:
        if exc.code == _REQUEST_TIMEOUT or "timed out" in exc.message:
            raise ToolFailed(f"Tool {name} timed out", transient=True) from exc
        raise ProtectClientError(redact(f"Tool {name} failed: {exc.message}")) from exc
    except Exception as exc:
        raise ProtectClientError(redact(f"Tool {name} failed: {exc}")) from exc
    if not isinstance(result, CallToolResult):
        raise ProtectClientError(f"Tool {name} returned an unexpected result")
    if result.is_error:
        raise _tool_failure(name, _result_text(result))
    if result.structured_content is not None:
        return _unwrap_result(result.structured_content)
    text = _result_text(result)
    try:
        return _unwrap_result(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ProtectClientError(
            f"Tool {name} returned non-JSON text: {redact(text)[:500]}"
        ) from exc


def _tool_failure(name: str, text: str) -> ToolFailed:
    """Read the server's `{"error", "details": {"status", ...}}` shape, or plain text."""
    message = text.strip() or f"Tool {name} failed"
    status: int | None = None
    two_factor = False
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        message = payload["error"]
        details = payload.get("details")
        if isinstance(details, dict):
            raw_status = details.get("status")
            if isinstance(raw_status, int) and not isinstance(raw_status, bool):
                status = raw_status
            two_factor = details.get("needsTwoFactor") is True
    if is_download_limit(message):
        return ToolFailed(message, status=status, transient=False)
    return ToolFailed(message, status=status, needs_two_factor=two_factor)


def _explain(exc: ToolFailed) -> str:
    message = str(exc)
    if exc.needs_two_factor:
        message += (
            " This CLI cannot enter a second factor. Log in once with the MCP server's own "
            "login steps so it caches a session, or use a local view-only account without 2FA."
        )
    elif exc.status in {401, 403}:
        message += " Check the UniFi Protect account and its permissions."
    elif is_download_limit(message):
        message += " Try --channel 1 or 2, a shorter --chunk-minutes, or a higher limit."
    return message


def _place_chunk(info: ExportInfo, dest: Path, start: datetime, end: datetime) -> FootageChunk:
    """Move the server's file into place and record its timing next to it."""
    partial = _partial_path(dest)
    meta: dict[str, object] = {
        "start": _iso(start),
        "end": _iso(end),
        "media_start": None if info.media_start is None else _iso_ms(info.media_start),
        "has_audio": info.has_audio,
    }
    side = sidecar_path(dest)
    side_partial = side.with_name(side.name + ".partial")
    side_partial.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    os.replace(side_partial, side)
    os.replace(partial, dest)
    return FootageChunk(
        path=dest,
        start=start,
        end=end,
        skipped=False,
        media_start=info.media_start,
        has_audio=info.has_audio,
    )


def _partial_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".partial")


def _same_file(reported: Path, expected: Path) -> bool:
    try:
        return reported.resolve() == expected.resolve()
    except OSError:
        return False


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


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _parse_iso(text: str) -> datetime:
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return _as_utc(datetime.fromisoformat(raw))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetimes must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _as_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_ms(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stamp(value: datetime) -> str:
    return _as_utc(value).strftime("%Y%m%dT%H%M%SZ")
