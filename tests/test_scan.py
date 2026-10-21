"""Scan pipeline with a mocked exporter, loader, and scorer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from protect_audio_search.classifier import DEFAULT_TARGET_CLASSES, FrameScore
from protect_audio_search.events import EventConfig
from protect_audio_search.footage import FootageChunk
from protect_audio_search.scan import run_scan

START = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)


def _frames() -> list[FrameScore]:
    return [
        FrameScore(time_seconds=0.0, scores={"Engine": 0.9}),
        FrameScore(time_seconds=0.48, scores={"Engine": 0.8}),
        FrameScore(time_seconds=0.96, scores={"Engine": 0.7}),
    ]


def test_run_scan_times_frames_from_each_chunk(tmp_path: Path) -> None:
    second = START + timedelta(minutes=10)
    end = START + timedelta(minutes=20)
    chunks = [
        FootageChunk(path=tmp_path / "a.mp4", start=START, end=second, skipped=False),
        FootageChunk(path=tmp_path / "b.mp4", start=second, end=end, skipped=False),
    ]
    progress: list[tuple[int, int]] = []

    def exporter(**_kwargs: object) -> list[FootageChunk]:
        return chunks

    def load_audio(path: Path, *, camera: str, chunk_label: str) -> str:
        assert camera == "Street"
        assert chunk_label == path.name
        return path.name

    scanned = run_scan(
        camera="Street",
        start=START,
        end=end,
        output_dir=tmp_path,
        event_config=EventConfig(),
        chunk_minutes=10,
        exporter=exporter,
        load_audio=load_audio,
        score_audio=lambda _waveform: _frames(),
        progress=lambda index, count, _start, _end: progress.append((index, count)),
    )
    assert progress == [(1, 2), (2, 2)]
    assert len(scanned.events) == 2
    assert scanned.chunks == chunks
    assert scanned.events[0].start == START
    assert scanned.events[0].camera == "Street"
    assert scanned.events[1].start == second
    assert run_scan.__kwdefaults__ is not None
    assert run_scan.__kwdefaults__["class_names"] == DEFAULT_TARGET_CLASSES


def test_silent_chunk_is_skipped(tmp_path: Path) -> None:
    end = START + timedelta(minutes=10)
    chunks = [FootageChunk(path=tmp_path / "quiet.mp4", start=START, end=end, skipped=False)]

    def refuse_score(_waveform: object) -> list[FrameScore]:
        raise AssertionError("silent audio must not be scored")

    scanned = run_scan(
        camera="Street",
        start=START,
        end=end,
        output_dir=tmp_path,
        event_config=EventConfig(),
        exporter=lambda **_kwargs: chunks,
        load_audio=lambda *_args, **_kwargs: None,
        score_audio=refuse_score,
    )
    assert scanned.events == []
