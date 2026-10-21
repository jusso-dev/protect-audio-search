"""Trim local footage chunks into review clips and JPEG snapshots.

Cuts stay on this machine. Nothing is uploaded.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from protect_audio_search.audio import AudioError, FfmpegMissingError, has_audio_stream
from protect_audio_search.events import DetectionEvent
from protect_audio_search.footage import FootageChunk
from protect_audio_search.times import format_utc

DEFAULT_PADDING_SECONDS = 5.0


@dataclass(frozen=True)
class EventMedia:
    clip: Path | None
    snapshots: tuple[Path, ...]


def review_directory(output_dir: Path, camera: str, start: datetime) -> Path:
    """One folder per scan so clips from different runs stay apart."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", camera).strip("._") or "camera"
    stamp = format_utc(start).replace("-", "").replace(":", "")
    return output_dir / f"{stamp}-{safe}"


def capture_events(
    events: Sequence[DetectionEvent],
    chunks: Sequence[FootageChunk],
    *,
    run_dir: Path,
    padding_seconds: float = DEFAULT_PADDING_SECONDS,
    write_clips: bool = False,
    snapshot_count: int = 0,
) -> list[EventMedia]:
    """Write a clip and snapshots for each event. Times may cross chunk boundaries."""
    if padding_seconds < 0:
        raise ValueError("clip padding must be zero or positive")
    if snapshot_count < 0:
        raise ValueError("snapshot count must be zero or positive")
    return [
        _capture_one(
            event,
            chunks,
            run_dir=run_dir,
            index=index,
            padding_seconds=padding_seconds,
            write_clips=write_clips,
            snapshot_count=snapshot_count,
        )
        for index, event in enumerate(events, start=1)
    ]


def delete_chunks(chunks: Sequence[FootageChunk]) -> int:
    """Delete exported footage files. Clips and snapshots are left in place."""
    removed = 0
    for chunk in chunks:
        if chunk.path.is_file():
            chunk.path.unlink()
            removed += 1
    return removed


def _capture_one(
    event: DetectionEvent,
    chunks: Sequence[FootageChunk],
    *,
    run_dir: Path,
    index: int,
    padding_seconds: float,
    write_clips: bool,
    snapshot_count: int,
) -> EventMedia:
    clip: Path | None = None
    if write_clips:
        window = _clip_window(event, chunks, padding_seconds)
        segments = _segments(chunks, window[0], window[1])
        if not segments:
            raise AudioError(f"No footage covers event {index}")
        dest = run_dir / "clips" / f"{index:04d}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_clip(segments, dest)
        clip = dest
    snapshots = _write_snapshots(event, chunks, run_dir=run_dir, index=index, count=snapshot_count)
    return EventMedia(clip=clip, snapshots=snapshots)


def _clip_window(
    event: DetectionEvent,
    chunks: Sequence[FootageChunk],
    padding_seconds: float,
) -> tuple[datetime, datetime]:
    if not chunks:
        raise AudioError("No footage chunks are available to trim")
    padding = timedelta(seconds=padding_seconds)
    available_start = min(chunk.start for chunk in chunks)
    available_end = max(chunk.end for chunk in chunks)
    start = max(event.start - padding, available_start)
    end = min(event.end + padding, available_end)
    if end <= start:
        raise AudioError("Clip range is empty after padding is clamped to the footage")
    return start, end


def _segments(
    chunks: Sequence[FootageChunk],
    start: datetime,
    end: datetime,
) -> list[tuple[Path, float, float, bool]]:
    segments: list[tuple[Path, float, float, bool]] = []
    for chunk in sorted(chunks, key=lambda item: item.start):
        seg_start = max(start, chunk.start)
        seg_end = min(end, chunk.end)
        if seg_end <= seg_start:
            continue
        offset = (seg_start - chunk.start).total_seconds()
        duration = (seg_end - seg_start).total_seconds()
        segments.append((chunk.path, offset, duration, has_audio_stream(chunk.path)))
    return segments


def _write_clip(segments: Sequence[tuple[Path, float, float, bool]], dest: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="clip-", dir=dest.parent) as temp_name:
        temp = Path(temp_name)
        parts: list[Path] = []
        with_audio = all(segment[3] for segment in segments)
        for index, (source, offset, duration, _has_audio) in enumerate(segments, start=1):
            part = temp / f"part-{index:02d}.mp4"
            _cut(source, offset, duration, part, with_audio=with_audio)
            parts.append(part)
        if len(parts) == 1:
            parts[0].replace(dest)
            return
        listing = temp / "parts.txt"
        listing.write_text(
            "".join(f"file '{part.as_posix()}'\n" for part in parts),
            encoding="utf-8",
        )
        _ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                str(dest),
            ]
        )


def _write_snapshots(
    event: DetectionEvent,
    chunks: Sequence[FootageChunk],
    *,
    run_dir: Path,
    index: int,
    count: int,
) -> tuple[Path, ...]:
    if count == 0:
        return ()
    duration = (event.end - event.start).total_seconds()
    if duration <= 0:
        raise AudioError(f"Event {index} has no duration to snapshot")
    folder = run_dir / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for snap in range(count):
        when = event.start + timedelta(seconds=duration * (snap + 0.5) / count)
        source, offset = _locate(chunks, when)
        dest = folder / f"{index:04d}-{snap + 1:02d}.jpg"
        _ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{offset:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(dest),
            ]
        )
        paths.append(dest)
    return tuple(paths)


def _locate(chunks: Sequence[FootageChunk], when: datetime) -> tuple[Path, float]:
    ordered = sorted(chunks, key=lambda item: item.start)
    for chunk in ordered:
        if chunk.start <= when < chunk.end:
            return chunk.path, (when - chunk.start).total_seconds()
    if ordered and when == ordered[-1].end:
        last = ordered[-1]
        span = (last.end - last.start).total_seconds()
        return last.path, max(0.0, span - 0.04)
    raise AudioError(f"No footage covers {when.isoformat()}")


def _cut(source: Path, offset: float, duration: float, dest: Path, *, with_audio: bool) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-ss",
        f"{offset:.3f}",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if with_audio:
        command.extend(["-c:a", "aac"])
    else:
        command.append("-an")
    command.append(str(dest))
    _ffmpeg(command)


def _ffmpeg(command: list[str]) -> None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=120)
    except FileNotFoundError as exc:
        raise FfmpegMissingError(
            "ffmpeg is not on PATH, so no footage was exported. "
            "Install ffmpeg (macOS: `brew install ffmpeg`, "
            "Debian/Ubuntu: `sudo apt install ffmpeg`)."
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError(f"ffmpeg failed: {detail or 'unknown error'}")
