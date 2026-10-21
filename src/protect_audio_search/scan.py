"""Run a scan: export chunks, score audio, merge events.

Callers can pass `exporter`, `load_audio`, and `score_audio` in tests. The CLI
uses the real MCP, ffmpeg, and YAMNet implementations.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from protect_audio_search.audio import load_chunk_audio
from protect_audio_search.classifier import DEFAULT_TARGET_CLASSES, FrameScore, classify
from protect_audio_search.events import DetectionEvent, EventConfig, ScoreFrame, detect_events
from protect_audio_search.footage import FootageChunk, export_footage, split_range
from protect_audio_search.times import format_zoned

Progress = Callable[[int, int, datetime, datetime], None]
Exporter = Callable[..., list[FootageChunk]]
AudioLoader = Callable[..., object | None]
Scorer = Callable[[object], list[FrameScore]]


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


def run_scan(
    *,
    camera: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    event_config: EventConfig,
    class_names: Sequence[str] = DEFAULT_TARGET_CLASSES,
    chunk_minutes: int = 10,
    command: str = "npx",
    args: Sequence[str] = ("-y", "@mgcrea/mcp-unifi-protect"),
    exporter: Exporter | None = None,
    load_audio: AudioLoader | None = None,
    score_audio: Scorer | None = None,
    progress: Progress = default_progress,
) -> ScanOutput:
    """Export each chunk, classify target classes, and return events plus chunks."""
    planned = plan_chunks(start, end, chunk_minutes)
    if exporter is None:
        chunks = asyncio.run(
            export_footage(
                camera=camera,
                start=start,
                end=end,
                output_dir=output_dir,
                command=command,
                args=args,
                chunk_minutes=chunk_minutes,
            )
        )
    else:
        chunks = exporter(
            camera=camera,
            start=start,
            end=end,
            output_dir=output_dir,
            chunk_minutes=chunk_minutes,
        )
    read_audio = load_audio or _load_audio
    score = score_audio or _score_with_yamnet(class_names)
    frames: list[ScoreFrame] = []
    for planned_chunk, chunk in zip(planned, chunks, strict=True):
        progress(planned_chunk.index, planned_chunk.count, chunk.start, chunk.end)
        waveform = read_audio(chunk.path, camera=camera, chunk_label=chunk.path.name)
        if waveform is None:
            continue
        for frame in score(waveform):
            frames.append(
                ScoreFrame(
                    time=chunk.start + timedelta(seconds=frame.time_seconds),
                    scores=frame.scores,
                )
            )
    return ScanOutput(
        events=detect_events(frames, camera=camera, config=event_config),
        chunks=chunks,
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
