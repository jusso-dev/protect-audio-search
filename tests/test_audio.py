"""Audio extraction against short ffmpeg-generated fixtures."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from protect_audio_search.audio import (
    SAMPLE_RATE,
    FfmpegMissingError,
    load_chunk_audio,
    media_duration_seconds,
    prepare_scan,
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", *args], check=True, capture_output=True)


def _with_audio(path: Path) -> None:
    _ffmpeg(
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=0.5:sample_rate=48000",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=32x32:rate=5:duration=0.5",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    )


def _video_only(path: Path) -> None:
    _ffmpeg(
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=32x32:rate=5:duration=0.5",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )


def test_audio_chunk_length_matches_duration(tmp_path: Path) -> None:
    path = tmp_path / "with-audio.mp4"
    _with_audio(path)
    samples = load_chunk_audio(path, camera="Street", chunk_label="1100-1110")
    assert samples is not None
    expected = media_duration_seconds(path) * SAMPLE_RATE
    assert samples.dtype.name == "float32"
    assert abs(len(samples) - expected) <= SAMPLE_RATE * 0.1
    assert float(samples.max()) <= 1.0
    assert float(samples.min()) >= -1.0


def test_video_only_chunk_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "silent.mp4"
    _video_only(path)
    with caplog.at_level(logging.WARNING):
        samples = load_chunk_audio(path, camera="Lane", chunk_label=path.name)
    assert samples is None
    assert "Lane" in caplog.text
    assert path.name in caplog.text
    assert "no audio stream" in caplog.text


def test_missing_ffmpeg_raises_before_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("protect_audio_search.audio.shutil.which", lambda _name: None)
    started = False

    def start_export() -> None:
        nonlocal started
        started = True

    with pytest.raises(FfmpegMissingError, match="no footage was exported"):
        prepare_scan(start_export)
    assert started is False
