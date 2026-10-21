"""Turn an exported video chunk into 16 kHz mono audio for YAMNet.

YAMNet expects a mono float waveform in [-1, 1] at 16 kHz. Cameras with no
microphone are skipped with a warning. `prepare_scan` checks ffmpeg is on
PATH before any footage export starts.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
_INSTALL_HINT = (
    "Install ffmpeg, which includes ffprobe "
    "(macOS: `brew install ffmpeg`, Debian/Ubuntu: `sudo apt install ffmpeg`)."
)

logger = logging.getLogger(__name__)


class AudioError(Exception):
    """ffmpeg or ffprobe failed on a chunk."""


class FfmpegMissingError(AudioError):
    """ffmpeg or ffprobe is not on PATH."""


def prepare_scan(start_export: Callable[[], None] | None = None) -> None:
    """Check ffmpeg, then run `start_export`. Nothing is exported if the check fails."""
    ensure_ffmpeg()
    if start_export is not None:
        start_export()


def ensure_ffmpeg() -> None:
    """Raise before a scan when ffmpeg or ffprobe is missing."""
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if not missing:
        return
    joined = " and ".join(missing)
    raise FfmpegMissingError(
        f"{joined} is not on PATH, so no footage was exported. {_INSTALL_HINT}"
    )


def load_chunk_audio(path: Path, *, camera: str, chunk_label: str) -> NDArray[np.float32] | None:
    """Return 16 kHz mono float32 samples, or None when the chunk has no audio."""
    if not has_audio_stream(path):
        logger.warning(
            "Camera %s chunk %s has no audio stream; skipping",
            camera,
            chunk_label,
        )
        return None
    return extract_pcm(path)


def has_audio_stream(path: Path) -> bool:
    """True when ffprobe reports at least one audio stream."""
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return bool(result.stdout.strip())


def extract_pcm(path: Path) -> NDArray[np.float32]:
    """Decode `path` to 16 kHz mono float32 PCM in [-1, 1]."""
    result = _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    if not result.stdout:
        raise AudioError(f"ffmpeg produced no audio for {path.name}")
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    return np.clip(samples, -1.0, 1.0)


def media_duration_seconds(path: Path) -> float:
    """Container duration in seconds."""
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AudioError(f"ffprobe did not report a duration for {path.name}") from exc


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=120)
    except FileNotFoundError as exc:
        raise FfmpegMissingError(
            f"{command[0]} is not on PATH, so no footage was exported. {_INSTALL_HINT}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError(f"{command[0]} failed for the chunk: {detail or 'unknown error'}")
    return result
