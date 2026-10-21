"""Full scan through the mock MCP server. No Protect console and no LLM."""

from __future__ import annotations

import math
import sys
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from protect_audio_search.events import EventConfig
from protect_audio_search.scan import run_scan

SERVER = Path(__file__).resolve().parent / "mock_protect_server.py"
ENGINE = Path(__file__).resolve().parent / "fixtures" / "engine_cc0.wav"
START = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)


def _synth(path: Path, kind: str) -> None:
    """Write a 1 second 16 kHz mono wav. `silence` is zero. `squeal` is a sweep."""
    rate = 16_000
    count = rate
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        for index in range(count):
            if kind == "silence":
                sample = 0
            else:
                position = index / rate
                frequency = 2_000 + 4_000 * position
                sample = int(16_000 * math.sin(2 * math.pi * frequency * position))
            frames += int(sample).to_bytes(2, "little", signed=True)
        handle.writeframes(frames)


def _scan(tmp_path: Path, wav: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setenv("MOCK_CALL_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("MOCK_EXPORT_DIR", str(tmp_path / "export"))
    monkeypatch.setenv("MOCK_MEDIA_WAV", str(wav))
    monkeypatch.delenv("MOCK_FAIL_FIRST", raising=False)
    monkeypatch.delenv("MOCK_PERMANENT_ERROR", raising=False)
    scanned = run_scan(
        camera="Street",
        start=START,
        end=START + timedelta(minutes=10),
        output_dir=tmp_path / "chunks",
        event_config=EventConfig(),
        command=sys.executable,
        args=[str(SERVER)],
    )
    assert scanned.chunks
    return [event.peak_class for event in scanned.events]


def test_engine_fixture_reports_motorcycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _scan(tmp_path, ENGINE, monkeypatch) == ["Motorcycle"]


def test_silence_fixture_reports_no_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "silence.wav"
    _synth(wav, "silence")
    assert _scan(tmp_path, wav, monkeypatch) == []


def test_synthetic_squeal_scans_without_a_required_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "squeal.wav"
    _synth(wav, "squeal")
    # The sweep is not a recording of tyre squeal. Any result is acceptable.
    _scan(tmp_path, wav, monkeypatch)
