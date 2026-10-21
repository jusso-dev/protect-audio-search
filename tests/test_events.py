"""Table-driven tests for event merging, thresholds, and chunk boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from protect_audio_search.cli import build_parser
from protect_audio_search.events import (
    EventConfig,
    ScoreFrame,
    detect_events,
    event_config_from_args,
)

UTC = UTC
T0 = datetime(2026, 10, 20, 11, 0, tzinfo=UTC)


def _frame(offset: float, **scores: float) -> ScoreFrame:
    return ScoreFrame(time=T0 + timedelta(seconds=offset), scores=scores)


def test_merges_nearby_frames_and_drops_short_blips() -> None:
    frames = [
        _frame(0.0, Engine=0.9),
        _frame(0.48, Engine=0.4),
        _frame(0.96, Engine=0.5),
        _frame(20.0, Engine=0.95),
    ]
    events = detect_events(
        frames,
        camera="Street",
        config=EventConfig(threshold=0.3, min_duration_seconds=1.5, gap_seconds=2),
    )
    assert len(events) == 1
    event = events[0]
    assert event.camera == "Street"
    assert event.start == T0
    assert event.end == T0 + timedelta(seconds=0.96 + 0.975)
    assert event.peak_class == "Engine"
    assert event.peak_score == pytest.approx(0.9)
    assert event.mean_score == pytest.approx((0.9 + 0.4 + 0.5) / 3)
    assert event.contributing_classes == ("Engine",)


def test_chunk_boundary_within_gap_is_one_event() -> None:
    frames = [
        _frame(0.0, Engine=0.8),
        _frame(0.48, Engine=0.8),
        _frame(0.96, Engine=0.8),
        # Next chunk starts just inside the gap.
        _frame(2.4, Engine=0.6),
        _frame(2.88, Engine=0.7),
    ]
    events = detect_events(frames, camera="Lane", config=EventConfig(min_duration_seconds=1.5))
    assert len(events) == 1
    assert events[0].start == T0
    assert events[0].end == T0 + timedelta(seconds=2.88 + 0.975)
    assert events[0].contributing_classes == ("Engine",)


def test_gap_larger_than_tolerance_splits_events() -> None:
    frames = [
        _frame(0.0, Engine=0.9),
        _frame(0.48, Engine=0.9),
        _frame(0.96, Engine=0.9),
        _frame(10.0, Engine=0.9),
        _frame(10.48, Engine=0.9),
        _frame(10.96, Engine=0.9),
    ]
    events = detect_events(frames, camera="Street", config=EventConfig())
    assert len(events) == 2
    assert events[1].start == T0 + timedelta(seconds=10)


def test_raising_threshold_never_increases_event_count() -> None:
    frames = [
        _frame(0.0, Engine=0.2, Motorcycle=0.9),
        _frame(0.48, Engine=0.4, Motorcycle=0.1),
        _frame(0.96, Engine=0.95, Motorcycle=0.2),
        _frame(1.44, Engine=0.5, Motorcycle=0.0),
        _frame(8.0, Engine=0.35, Motorcycle=0.0),
        _frame(8.48, Engine=0.35, Motorcycle=0.0),
        _frame(8.96, Engine=0.35, Motorcycle=0.0),
    ]
    counts = [
        len(detect_events(frames, camera="Street", config=EventConfig(threshold=threshold)))
        for threshold in (0.1, 0.3, 0.5, 0.8, 1.1)
    ]
    assert counts == sorted(counts, reverse=True)


def test_per_class_threshold_and_contributing_classes() -> None:
    frames = [
        _frame(0.0, Engine=0.4, Motorcycle=0.9),
        _frame(0.48, Engine=0.45, Motorcycle=0.2),
        _frame(0.96, Engine=0.5, Motorcycle=0.1),
    ]
    events = detect_events(
        frames,
        camera="Street",
        config=EventConfig(
            threshold=0.3,
            class_thresholds={"Motorcycle": 0.95},
            min_duration_seconds=1.5,
        ),
    )
    assert len(events) == 1
    assert events[0].peak_class == "Engine"
    assert events[0].contributing_classes == ("Engine",)


def test_scan_help_documents_detection_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["scan", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "--threshold" in text
    assert "--class-threshold" in text
    assert "--smooth-frames" in text
    assert "--gap" in text
    assert "--min-duration" in text


def test_class_threshold_flag_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "scan",
            "--camera",
            "Street",
            "--from",
            "2026-10-20 21:00",
            "--to",
            "2026-10-21 02:00",
            "--threshold",
            "0.4",
            "--class-threshold",
            "Engine=0.2",
            "--gap",
            "3",
            "--smooth-frames",
            "3",
        ]
    )
    config = event_config_from_args(args)
    assert config.threshold == pytest.approx(0.4)
    assert config.threshold_for("Engine") == pytest.approx(0.2)
    assert config.threshold_for("Motorcycle") == pytest.approx(0.4)
    assert config.gap_seconds == pytest.approx(3)
    assert config.smooth_frames == 3
