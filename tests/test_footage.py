"""Chunked export against the mock Protect MCP server."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from protect_audio_search.footage import (
    Camera,
    FootageChunk,
    ProtectClientError,
    export_footage,
    export_footage_or_exit,
    resolve_camera,
    split_range,
)

SERVER = Path(__file__).with_name("mock_protect_server.py")
START = datetime(2026, 10, 20, 11, 0, tzinfo=UTC)
END = datetime(2026, 10, 20, 11, 25, tzinfo=UTC)


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["MOCK_CALL_LOG"] = str(tmp_path / "calls.jsonl")
    env["MOCK_EXPORT_DIR"] = str(tmp_path / "server-exports")
    env.update(extra)
    return env


def _calls(tmp_path: Path) -> list[dict[str, object]]:
    log = tmp_path / "calls.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def _export(tmp_path: Path, **extra: object) -> list[FootageChunk]:
    camera = extra.pop("camera", "Street")
    env = extra.pop("env", _env(tmp_path))
    assert isinstance(camera, str)
    assert isinstance(env, dict)
    return asyncio.run(
        export_footage(
            camera=camera,
            start=START,
            end=END,
            output_dir=tmp_path / "chunks",
            command=sys.executable,
            args=[str(SERVER)],
            env=env,
            retries=3,
            retry_backoff_seconds=0,
        )
    )


def test_split_range_25_minutes_is_three_chunks() -> None:
    ten = datetime(2026, 10, 20, 11, 10, tzinfo=UTC)
    twenty = datetime(2026, 10, 20, 11, 20, tzinfo=UTC)
    assert split_range(START, END, 10) == [(START, ten), (ten, twenty), (twenty, END)]


def test_export_calls_three_chunks_with_utc_boundaries(tmp_path: Path) -> None:
    chunks = _export(tmp_path)
    assert len(chunks) == 3
    assert [chunk.skipped for chunk in chunks] == [False, False, False]
    calls = _calls(tmp_path)
    assert [(call["start"], call["end"]) for call in calls] == [
        ("2026-10-20T11:00:00Z", "2026-10-20T11:10:00Z"),
        ("2026-10-20T11:10:00Z", "2026-10-20T11:20:00Z"),
        ("2026-10-20T11:20:00Z", "2026-10-20T11:25:00Z"),
    ]
    assert {call["cameraId"] for call in calls} == {"cam-1"}
    for chunk in chunks:
        assert chunk.path.is_file()
        assert chunk.path.stat().st_size > 0
        assert chunk.path.parent.name == "cam-1"


def test_rerun_skips_existing_chunk_files(tmp_path: Path) -> None:
    first = _export(tmp_path)
    log = tmp_path / "calls.jsonl"
    log.unlink()
    second = _export(tmp_path)
    assert [chunk.skipped for chunk in second] == [True, True, True]
    assert [chunk.path for chunk in second] == [chunk.path for chunk in first]
    assert _calls(tmp_path) == []


def test_unknown_camera_lists_available(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        export_footage_or_exit(
            camera="Nope",
            start=START,
            end=END,
            output_dir=tmp_path / "chunks",
            command=sys.executable,
            args=[str(SERVER)],
            env=_env(tmp_path),
            retry_backoff_seconds=0,
        )
    assert caught.value.code == 1
    error = capsys.readouterr().err
    assert "Unknown camera" in error
    assert "Street" in error
    assert "Lane" in error
    assert _calls(tmp_path) == []


def test_missing_server_command_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        export_footage_or_exit(
            camera="Street",
            start=START,
            end=END,
            output_dir=tmp_path / "chunks",
            command="definitely-missing-mcp-bin",
            args=["--help"],
            env=_env(tmp_path),
        )
    assert caught.value.code == 1
    error = capsys.readouterr().err
    assert "definitely-missing-mcp-bin" in error
    assert "PATH" in error


def test_tool_error_is_readable_and_redacts_secrets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        export_footage_or_exit(
            camera="Street",
            start=START,
            end=END,
            output_dir=tmp_path / "chunks",
            command=sys.executable,
            args=[str(SERVER)],
            env=_env(tmp_path, MOCK_PERMANENT_ERROR="password=supersecret rejected"),
            retries=3,
            retry_backoff_seconds=0,
        )
    assert caught.value.code == 1
    error = capsys.readouterr().err
    ours = [line for line in error.splitlines() if line.startswith("Export failed")]
    assert ours
    assert "supersecret" not in ours[0]
    assert "[redacted]" in ours[0]
    assert len(_calls(tmp_path)) == 1


def test_transient_export_error_is_retried(tmp_path: Path) -> None:
    chunks = _export(tmp_path, env=_env(tmp_path, MOCK_FAIL_FIRST="1"))
    assert len(chunks) == 3
    assert all(chunk.path.is_file() for chunk in chunks)
    # First chunk fails once then succeeds. The other two succeed on the first try.
    assert len(_calls(tmp_path)) == 4


def test_duplicate_camera_name_asks_for_id() -> None:
    cameras = [Camera("a", "Street"), Camera("b", "Street")]
    with pytest.raises(ProtectClientError, match="camera id"):
        resolve_camera(cameras, "street")
