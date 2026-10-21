"""Turn per-frame class scores into discrete events.

Frames whose score passes a threshold are merged when the gap between them is
short, including across footage chunk boundaries. Short blips are dropped.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

DEFAULT_THRESHOLD = 0.3
DEFAULT_GAP_SECONDS = 2.0
DEFAULT_MIN_DURATION_SECONDS = 1.5
DEFAULT_WINDOW_SECONDS = 0.975


@dataclass(frozen=True)
class EventConfig:
    """Detection parameters. The CLI exposes the same fields."""

    threshold: float = DEFAULT_THRESHOLD
    class_thresholds: Mapping[str, float] = field(default_factory=dict)
    smooth_frames: int = 1
    gap_seconds: float = DEFAULT_GAP_SECONDS
    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS
    window_seconds: float = DEFAULT_WINDOW_SECONDS

    def threshold_for(self, class_name: str) -> float:
        return self.class_thresholds.get(class_name, self.threshold)


@dataclass(frozen=True)
class ScoreFrame:
    """One classifier window. `time` is the window start in UTC."""

    time: datetime
    scores: Mapping[str, float]


@dataclass(frozen=True)
class DetectionEvent:
    camera: str
    start: datetime
    end: datetime
    peak_class: str
    peak_score: float
    mean_score: float
    contributing_classes: tuple[str, ...]


def detect_events(
    frames: Sequence[ScoreFrame],
    *,
    camera: str,
    config: EventConfig | None = None,
) -> list[DetectionEvent]:
    """Merge active frames into events. Higher thresholds never add events."""
    settings = config or EventConfig()
    _validate(settings)
    ordered = sorted(frames, key=lambda frame: _as_utc(frame.time))
    if not ordered:
        return []
    class_names = list(ordered[0].scores)
    smoothed = {
        name: _smooth(
            [float(frame.scores.get(name, 0.0)) for frame in ordered],
            settings.smooth_frames,
        )
        for name in class_names
    }
    active_indexes = [
        index
        for index in range(len(ordered))
        if any(smoothed[name][index] >= settings.threshold_for(name) for name in class_names)
    ]
    events: list[DetectionEvent] = []
    for group in _group_indexes(active_indexes, ordered, settings.gap_seconds):
        event = _event_from_group(group, ordered, smoothed, camera, settings)
        if event is not None:
            events.append(event)
    return events


def add_event_arguments(parser: argparse.ArgumentParser) -> None:
    """Add detection flags to a parser. Names match `EventConfig`."""
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Score threshold for every class (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--class-threshold",
        action="append",
        default=[],
        metavar="NAME=SCORE",
        help="Override the threshold for one class. Repeat for more than one class",
    )
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=1,
        help="Trailing moving-average length in frames. 1 turns smoothing off",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_GAP_SECONDS,
        dest="gap_seconds",
        help=f"Merge frames separated by at most this many seconds (default {DEFAULT_GAP_SECONDS})",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION_SECONDS,
        dest="min_duration_seconds",
        help=(
            f"Drop events shorter than this many seconds (default {DEFAULT_MIN_DURATION_SECONDS})"
        ),
    )


def event_config_from_args(args: argparse.Namespace) -> EventConfig:
    overrides: dict[str, float] = {}
    for item in args.class_threshold:
        name, separator, raw_score = str(item).partition("=")
        if separator != "=" or not name:
            raise ValueError(f"Class threshold must be NAME=SCORE, got {item!r}")
        overrides[name] = float(raw_score)
    return EventConfig(
        threshold=float(args.threshold),
        class_thresholds=overrides,
        smooth_frames=int(args.smooth_frames),
        gap_seconds=float(args.gap_seconds),
        min_duration_seconds=float(args.min_duration_seconds),
    )


def _event_from_group(
    indexes: list[int],
    frames: Sequence[ScoreFrame],
    smoothed: dict[str, list[float]],
    camera: str,
    config: EventConfig,
) -> DetectionEvent | None:
    start = _as_utc(frames[indexes[0]].time)
    # The last window still covers `window_seconds` of audio.
    end = _as_utc(frames[indexes[-1]].time) + timedelta(seconds=config.window_seconds)
    if (end - start).total_seconds() < config.min_duration_seconds:
        return None
    contributing = tuple(
        sorted(
            name
            for name in smoothed
            if any(smoothed[name][index] >= config.threshold_for(name) for index in indexes)
        )
    )
    peak_class = ""
    peak_score = -1.0
    for index in indexes:
        for name in contributing:
            score = smoothed[name][index]
            if score > peak_score or (score == peak_score and name < peak_class):
                peak_score = score
                peak_class = name
    peak_values = [smoothed[peak_class][index] for index in indexes]
    return DetectionEvent(
        camera=camera,
        start=start,
        end=end,
        peak_class=peak_class,
        peak_score=peak_score,
        mean_score=sum(peak_values) / len(peak_values),
        contributing_classes=contributing,
    )


def _group_indexes(
    indexes: list[int],
    frames: Sequence[ScoreFrame],
    gap_seconds: float,
) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for index in indexes:
        if not current:
            current = [index]
            continue
        gap = (_as_utc(frames[index].time) - _as_utc(frames[current[-1]].time)).total_seconds()
        if gap <= gap_seconds:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    if current:
        groups.append(current)
    return groups


def _smooth(values: list[float], width: int) -> list[float]:
    if width <= 1:
        return values
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - width + 1)
        window = values[start : index + 1]
        smoothed.append(sum(window) / len(window))
    return smoothed


def _validate(config: EventConfig) -> None:
    if config.smooth_frames < 1:
        raise ValueError("smooth_frames must be at least 1")
    if config.gap_seconds < 0:
        raise ValueError("gap_seconds must be zero or positive")
    if config.min_duration_seconds < 0:
        raise ValueError("min_duration_seconds must be zero or positive")
    if config.window_seconds <= 0:
        raise ValueError("window_seconds must be positive")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("frame times must be timezone-aware")
    return value
