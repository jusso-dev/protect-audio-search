"""The export client against the real server's contract, through the mock."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError

from protect_audio_search.footage import (
    SERVER_PACKAGE,
    FailedChunk,
    FootageChunk,
    ProtectClientError,
    ToolFailed,
    _call_tool,
    child_environment,
    export_footage,
    list_cameras,
    redact,
    server_command,
    sidecar_path,
)

SERVER = Path(__file__).with_name("mock_protect_server.py")
START = datetime(2026, 10, 20, 11, 0, tzinfo=UTC)
END = datetime(2026, 10, 20, 11, 20, tzinfo=UTC)


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["MOCK_CALL_LOG"] = str(tmp_path / "calls.jsonl")
    env["MOCK_EXPORT_DIR"] = str(tmp_path / "server-cache")
    env.update(extra)
    return env


def _calls(tmp_path: Path) -> list[dict[str, object]]:
    log = tmp_path / "calls.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def _export(tmp_path: Path, env: dict[str, str], **extra: object) -> list[FootageChunk]:
    return asyncio.run(
        export_footage(
            camera="Street",
            start=START,
            end=END,
            output_dir=tmp_path / "chunks",
            command=sys.executable,
            args=[str(SERVER)],
            env=env,
            retry_backoff_seconds=0,
            **extra,  # type: ignore[arg-type]
        )
    )


def test_export_writes_straight_to_the_chunk_and_leaves_no_server_copy(tmp_path: Path) -> None:
    chunks = _export(tmp_path, _env(tmp_path, MOCK_PREROLL_MS="700"), channel=2)
    calls = _calls(tmp_path)
    assert {call["channel"] for call in calls} == {2}
    assert all(str(call["savePath"]).endswith(".mp4.partial") for call in calls)
    assert all(Path(str(call["savePath"])).is_absolute() for call in calls)
    assert not (tmp_path / "server-cache").exists()
    for chunk in chunks:
        assert chunk.path.is_file()
        assert not chunk.path.with_name(chunk.path.name + ".partial").exists()
        assert chunk.media_start == chunk.start - timedelta(milliseconds=700)
        assert chunk.origin == chunk.media_start
        assert chunk.has_audio is True
        assert sidecar_path(chunk.path).is_file()


def test_resumed_chunk_keeps_starts_at_from_the_sidecar(tmp_path: Path) -> None:
    first = _export(tmp_path, _env(tmp_path, MOCK_PREROLL_MS="400"))
    second = _export(tmp_path, _env(tmp_path, MOCK_PREROLL_MS="400"))
    assert [chunk.skipped for chunk in second] == [True, True]
    assert [chunk.media_start for chunk in second] == [chunk.media_start for chunk in first]


def test_server_that_ignores_save_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProtectClientError, match="not the requested"):
        _export(tmp_path, _env(tmp_path, MOCK_IGNORE_SAVE_PATH="1"))
    assert not list((tmp_path / "chunks").rglob("*.mp4*"))


def test_chunk_over_the_download_limit_is_split(tmp_path: Path) -> None:
    progress: list[tuple[int, int]] = []
    chunks = _export(
        tmp_path,
        _env(tmp_path, MOCK_MAX_SECONDS="300"),
        progress=lambda index, count, _s, _e, _cached: progress.append((index, count)),
    )
    spans = [(chunk.end - chunk.start).total_seconds() for chunk in chunks]
    assert spans == [300.0] * 4
    assert chunks[0].start == START
    assert chunks[-1].end == END
    assert {count for _index, count in progress} == {2}


def test_failed_chunk_is_reported_and_the_run_continues(tmp_path: Path) -> None:
    failures: list[FailedChunk] = []
    ready: list[FootageChunk] = []
    chunks = _export(
        tmp_path,
        _env(tmp_path, MOCK_FAIL_STARTS="2026-10-20T11:00:00Z"),
        on_failure=failures.append,
        on_chunk=ready.append,
    )
    assert [chunk.start for chunk in chunks] == [START + timedelta(minutes=10)]
    assert ready == chunks
    assert [failure.start for failure in failures] == [START]
    assert "no recording" in failures[0].error


def test_every_chunk_failing_stops_the_run(tmp_path: Path) -> None:
    failures: list[FailedChunk] = []
    with pytest.raises(ProtectClientError, match="in a row"):
        asyncio.run(
            export_footage(
                camera="Street",
                start=START,
                end=START + timedelta(hours=1),
                output_dir=tmp_path / "chunks",
                command=sys.executable,
                args=[str(SERVER)],
                env=_env(tmp_path, MOCK_PERMANENT_ERROR="Not found"),
                retry_backoff_seconds=0,
                on_failure=failures.append,
            )
        )
    assert len(failures) == 3


def test_camera_without_audio_is_refused_before_export(tmp_path: Path) -> None:
    cameras = json.dumps([{"id": "cam-1", "name": "Street", "recordsAudio": False}])
    with pytest.raises(ProtectClientError, match="does not record audio"):
        _export(tmp_path, _env(tmp_path, MOCK_CAMERAS=cameras), require_audio=True)
    assert _calls(tmp_path) == []


def test_unconfigured_server_names_the_variables(tmp_path: Path) -> None:
    with pytest.raises(ProtectClientError, match="UNIFI_PROTECT_HOST"):
        asyncio.run(
            list_cameras(
                command=sys.executable,
                args=[str(SERVER)],
                env=_env(tmp_path, MOCK_UNCONFIGURED="1"),
            )
        )


def test_camera_list_reads_audio_and_recording_fields(tmp_path: Path) -> None:
    cameras = asyncio.run(
        list_cameras(command=sys.executable, args=[str(SERVER)], env=_env(tmp_path))
    )
    assert cameras[0].records_audio is True
    assert cameras[0].recording_mode == "always"


def test_request_timeout_is_retryable() -> None:
    class TimingOut:
        async def call_tool(self, _name: str, _arguments: dict[str, object]) -> object:
            raise MCPError(code=-32001, message="Request 'tools/call' timed out")

    with pytest.raises(ToolFailed) as caught:
        asyncio.run(_call_tool(TimingOut(), "unifi_protect_export_video", {}))  # type: ignore[arg-type]
    assert caught.value.transient


def test_transient_status_comes_from_the_error_details() -> None:
    assert ToolFailed("Bad gateway", status=502).transient
    assert not ToolFailed("Camera 5001 not found", status=404).transient
    assert not ToolFailed("over the 200000000-byte limit", status=None, transient=False).transient


def test_redact_handles_json_bearer_urls_and_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNIFI_PROTECT_PASSWORD", "hunter2-literal")
    text = redact(
        '{"details":{"password":"hunter 2","url":"https://admin:pw@10.0.0.1/x"}} '
        "Authorization: Bearer abc.def.ghi token=xyz and hunter2-literal"
    )
    for secret in ("hunter 2", "admin:pw", "abc.def.ghi", "xyz", "hunter2-literal"):
        assert secret not in text
    assert '"password":"[redacted]"' in text


def test_child_environment_drops_other_service_keys() -> None:
    env = child_environment(
        {
            "PATH": "/bin",
            "HOME": "/home/me",
            "UNIFI_PROTECT_HOST": "10.0.0.1",
            "ANTHROPIC_API_KEY": "sk-ant-x",
            "AWS_SECRET_ACCESS_KEY": "aws",
            "HTTPS_PROXY": "http://proxy",
        }
    )
    assert env == {
        "PATH": "/bin",
        "HOME": "/home/me",
        "UNIFI_PROTECT_HOST": "10.0.0.1",
        "HTTPS_PROXY": "http://proxy",
    }


def test_server_command_is_pinned_and_overridable() -> None:
    assert server_command({}) == ("npx", ("-y", SERVER_PACKAGE))
    assert "@" in SERVER_PACKAGE.removeprefix("@")
    assert server_command({"PROTECT_AUDIO_SEARCH_MCP_COMMAND": "mcp-unifi-protect --x 'a b'"}) == (
        "mcp-unifi-protect",
        ("--x", "a b"),
    )
