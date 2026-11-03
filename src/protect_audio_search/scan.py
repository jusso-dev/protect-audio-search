"""Run a scan: export chunks, score audio, merge events.

The classifier is loaded before the first export, so a missing or bad model
fails in seconds, not after hours of footage. Each chunk is scored as soon as
it lands. One chunk that cannot be exported or decoded is recorded and the run
carries on.

Callers can pass `exporter`, `load_audio`, and `score_audio` in tests. The CLI
uses the real MCP, ffmpeg, and YAMNet implementations.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from protect_audio_search.audio import SAMPLE_RATE, AudioError, load_chunk_audio
from protect_audio_search.classifier import DEFAULT_TARGET_CLASSES, FrameScore, classify
from protect_audio_search.events import DetectionEvent, EventConfig, ScoreFrame, detect_events
from protect_audio_search.footage import (
    DEFAULT_CHANNEL,
    ExportProgress,
    FailedChunk,
    FootageChunk,
    export_footage,
    split_range,
)
from protect_audio_search.times import format_zoned

Progress = Callable[[int, int, datetime, datetime], None]
Exporter = Callable[..., list[FootageChunk]]
AudioLoader = Callable[..., object | None]
Scorer = Callable[[object], list[FrameScore]]
Warn = Callable[[str], None]

# A chunk whose audio is this much shorter than its range gets a warning.
SHORT_AUDIO_TOLERANCE_SECONDS = 5.0


@dataclass(frozen=True)
class PlannedChunk:
    index: int
    count: int
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ScanOutput:
    events: list[DetectionEvent]
    chunks: list[FootageChunk]
    failed: list[FailedChunk] = field(default_factory=list)
    silent: list[FootageChunk] = field(default_factory=list)
    short: list[FootageChunk] = field(default_factory=list)
    scored: int = 0
    planned: int = 0


def plan_chunks(start: datetime, end: datetime, chunk_minutes: int) -> list[PlannedChunk]:
    ranges = split_range(start, end, chunk_minutes)
    count = len(ranges)
    return [
        PlannedChunk(index=index, count=count, start=chunk_start, end=chunk_end)
        for index, (chunk_start, chunk_end) in enumerate(ranges, start=1)
    ]


def format_plan(chunks: Sequence[PlannedChunk], tz: ZoneInfo) -> str:
    lines = [f"Planned chunks: {len(chunks)}"]
    for chunk in chunks:
        lines.append(
            f"chunk {chunk.index}/{chunk.count} "
            f"{format_zoned(chunk.start, tz)} to {format_zoned(chunk.end, tz)}"
        )
    return "\n".join(lines)


def default_progress(index: int, count: int, start: datetime, end: datetime) -> None:
    print(f"chunk {index}/{count} {start.isoformat()} to {end.isoformat()}", file=sys.stderr)


def default_warn(message: str) -> None:
    print(message, file=sys.stderr)


def run_scan(
    *,
    camera: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    event_config: EventConfig,
    class_names: Sequence[str] = DEFAULT_TARGET_CLASSES,
    chunk_minutes: int = 10,
    channel: int = DEFAULT_CHANNEL,
    command: str | None = None,
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    require_audio: bool = False,
    exporter: Exporter | None = None,
    load_audio: AudioLoader | None = None,
    score_audio: Scorer | None = None,
    progress: Progress = default_progress,
    export_progress: ExportProgress | None = None,
    warn: Warn = default_warn,
) -> ScanOutput:
    """Export each chunk, classify target classes, and return events plus chunks."""
    planned = plan_chunks(start, end, chunk_minutes)
    # Load the model first. A bad cache should not cost a five-hour export.
    score = score_audio or _score_with_yamnet(class_names)
    read_audio = load_audio or _load_audio
    frames: list[ScoreFrame] = []
    chunks: list[FootageChunk] = []
    failed: list[FailedChunk] = []
    silent: list[FootageChunk] = []
    short: list[FootageChunk] = []
    scored = 0

    def index_of(chunk: FootageChunk) -> PlannedChunk:
        for item in planned:
            if item.start <= chunk.start < item.end:
                return item
        return planned[-1]

    def fail(item: FailedChunk) -> None:
        failed.append(item)
        warn(f"Chunk {item.start.isoformat()} to {item.end.isoformat()} failed: {item.error}")

    def process(chunk: FootageChunk) -> None:
        nonlocal scored
        chunks.append(chunk)
        if chunk.media_start is None:
            warn(
                f"Chunk {chunk.path.name} has no export start time; event times may be approximate."
            )
        planned_chunk = index_of(chunk)
        progress(planned_chunk.index, planned_chunk.count, chunk.start, chunk.end)
        if chunk.has_audio is False:
            silent.append(chunk)
            warn(f"Chunk {chunk.path.name} has no audio track. It was not scored.")
            return
        try:
            waveform = read_audio(chunk.path, camera=camera, chunk_label=chunk.path.name)
        except AudioError as exc:
            message = str(exc)
            if chunk.skipped:
                message += f" Delete {chunk.path} so the next run exports it again."
            fail(FailedChunk(chunk.start, chunk.end, message))
            return
        if waveform is None:
            # `load_chunk_audio` has already logged a warning naming the chunk.
            silent.append(chunk)
            return
        span = (chunk.end - chunk.start).total_seconds()
        heard = np.asarray(waveform).size / SAMPLE_RATE
        if heard < span - SHORT_AUDIO_TOLERANCE_SECONDS:
            short.append(chunk)
            warn(
                f"Chunk {chunk.path.name} has {heard:.0f} s of audio for a {span:.0f} s range. "
                "Protect may have no recording for part of it, so times in it are approximate."
            )
        available_end = min(chunk.end, chunk.origin + timedelta(seconds=heard))
        chunk_frames: list[ScoreFrame] = []
        for frame in score(waveform):
            when = chunk.origin + timedelta(seconds=frame.time_seconds)
            # The export opens on the keyframe before `start`. That lead-in belongs
            # to the previous chunk and would otherwise be counted twice.
            if when < chunk.start or when >= available_end:
                continue
            chunk_frames.append(ScoreFrame(time=when, scores=frame.scores, end=available_end))
        frames.extend(chunk_frames)
        scored += 1

    if exporter is None:
        asyncio.run(
            export_footage(
                camera=camera,
                start=start,
                end=end,
                output_dir=output_dir,
                command=command,
                args=args,
                env=env,
                chunk_minutes=chunk_minutes,
                channel=channel,
                require_audio=require_audio,
                warn=warn,
                progress=export_progress,
                on_chunk=process,
                on_failure=fail,
            )
        )
    else:
        for chunk in exporter(
            camera=camera,
            start=start,
            end=end,
            output_dir=output_dir,
            chunk_minutes=chunk_minutes,
        ):
            process(chunk)
    return ScanOutput(
        events=detect_events(frames, camera=camera, config=event_config),
        chunks=chunks,
        failed=failed,
        silent=silent,
        short=short,
        scored=scored,
        planned=len(planned),
    )


def _load_audio(path: Path, *, camera: str, chunk_label: str) -> object | None:
    return load_chunk_audio(path, camera=camera, chunk_label=chunk_label)


def _score_with_yamnet(class_names: Sequence[str]) -> Scorer:
    from protect_audio_search.classifier import load_classifier

    backend, class_map, selected = load_classifier(class_names=list(class_names))

    def score(waveform: object) -> list[FrameScore]:
        samples = np.asarray(waveform, dtype=np.float32)
        return classify(samples, selected, backend, class_map)

    return score
