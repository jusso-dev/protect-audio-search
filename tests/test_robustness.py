"""Edge cases found in review: timing, smoothing, DST, tails, and partial failures."""

from __future__ import annotations

import csv
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from protect_audio_search.audio import AudioError
from protect_audio_search.classifier import (
    CLASS_COUNT,
    HOP_SAMPLES,
    WINDOW_SAMPLES,
    ClassMap,
    FrameScore,
    classify,
)
from protect_audio_search.clips import capture_events, review_directory
from protect_audio_search.confirm import (
    ConfirmBudget,
    ConfirmError,
    ProviderReply,
    confirm_events,
    parse_confirmation,
)
from protect_audio_search.events import DetectionEvent, EventConfig, ScoreFrame, detect_events
from protect_audio_search.footage import FootageChunk
from protect_audio_search.output import ScanResult, write_csv
from protect_audio_search.scan import run_scan
from protect_audio_search.times import parse_user_time

T0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
SYDNEY = ZoneInfo("Australia/Sydney")


def _event(start: float, end: float) -> DetectionEvent:
    return DetectionEvent(
        camera="Street",
        start=T0 + timedelta(seconds=start),
        end=T0 + timedelta(seconds=end),
        peak_class="Engine",
        peak_score=0.9,
        mean_score=0.8,
        contributing_classes=("Engine",),
    )


def _video(path: Path, seconds: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}:sample_rate=48000",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=32x32:rate=10:duration={seconds}",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


# Times


def test_skipped_dst_time_is_refused() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        parse_user_time("2026-10-04 02:30", SYDNEY)


def test_repeated_dst_time_is_refused() -> None:
    with pytest.raises(ValueError, match="happens twice"):
        parse_user_time("2026-04-05 02:30", SYDNEY)


def test_explicit_offset_inside_a_dst_change_is_accepted() -> None:
    assert parse_user_time("2026-04-05T02:30+11:00", SYDNEY) == datetime(
        2026, 4, 4, 15, 30, tzinfo=UTC
    )


def test_sub_second_times_are_truncated() -> None:
    assert parse_user_time("2026-10-20T10:00:00.700Z", SYDNEY).microsecond == 0


# Classifier tail


def test_tail_of_a_chunk_is_scored() -> None:
    class Flat:
        def score_window(self, window: np.ndarray) -> np.ndarray:
            assert window.shape == (WINDOW_SAMPLES,)
            return np.full(CLASS_COUNT, 0.5, dtype=np.float32)

    names = [f"class-{index}" for index in range(CLASS_COUNT)]
    samples = np.zeros(WINDOW_SAMPLES + HOP_SAMPLES + 1000, dtype=np.float32)
    frames = classify(samples, ["class-0"], Flat(), ClassMap(names))
    assert [frame.time_seconds for frame in frames] == [0.0, 0.48, 0.96]


# Smoothing


def test_smoothing_does_not_cross_a_gap_in_audio() -> None:
    loud = [ScoreFrame(T0 + timedelta(seconds=0.48 * i), {"Engine": 0.95}) for i in range(8)]
    later = T0 + timedelta(minutes=20)
    quiet = [ScoreFrame(later + timedelta(seconds=0.48 * i), {"Engine": 0.0}) for i in range(8)]
    events = detect_events(loud + quiet, camera="Street", config=EventConfig(smooth_frames=8))
    assert all(event.start < later for event in events)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.5])
def test_out_of_range_threshold_is_refused(value: float) -> None:
    with pytest.raises(ValueError):
        detect_events([], camera="Street", config=EventConfig(threshold=value))


# Scan


def test_model_loads_before_any_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken_model(_names: object) -> object:
        raise AudioError("model is broken")

    def exporter(**_kwargs: object) -> list[FootageChunk]:
        raise AssertionError("export must not start before the model loads")

    monkeypatch.setattr("protect_audio_search.scan._score_with_yamnet", broken_model)
    with pytest.raises(AudioError, match="model is broken"):
        run_scan(
            camera="Street",
            start=T0,
            end=T0 + timedelta(minutes=10),
            output_dir=tmp_path,
            event_config=EventConfig(),
            exporter=exporter,
        )


def test_frames_use_starts_at_and_drop_the_lead_in(tmp_path: Path) -> None:
    chunk = FootageChunk(
        path=tmp_path / "a.mp4",
        start=T0,
        end=T0 + timedelta(minutes=10),
        skipped=False,
        media_start=T0 - timedelta(seconds=1),
    )
    seen: list[float] = []

    def score(_waveform: object) -> list[FrameScore]:
        frames = [FrameScore(time_seconds=0.48 * i, scores={"Engine": 0.9}) for i in range(10)]
        seen.extend(frame.time_seconds for frame in frames)
        return frames

    scanned = run_scan(
        camera="Street",
        start=T0,
        end=T0 + timedelta(minutes=10),
        output_dir=tmp_path,
        event_config=EventConfig(),
        exporter=lambda **_kwargs: [chunk],
        load_audio=lambda *_args, **_kwargs: np.zeros(16_000 * 600, dtype=np.float32),
        score_audio=score,
        progress=lambda *_args: None,
    )
    assert len(scanned.events) == 1
    # Offset 0.96 s in the file is 0.04 s before the chunk start, so it is dropped.
    assert scanned.events[0].start == T0 + timedelta(seconds=0.44)


def test_one_bad_chunk_is_recorded_and_the_rest_are_scored(tmp_path: Path) -> None:
    second = T0 + timedelta(minutes=10)
    chunks = [
        FootageChunk(path=tmp_path / "bad.mp4", start=T0, end=second, skipped=True),
        FootageChunk(
            path=tmp_path / "good.mp4",
            start=second,
            end=second + timedelta(minutes=10),
            skipped=False,
        ),
        FootageChunk(
            path=tmp_path / "mute.mp4",
            start=second,
            end=second + timedelta(minutes=10),
            skipped=False,
            has_audio=False,
        ),
    ]

    def load(path: Path, **_kwargs: object) -> object:
        if path.name == "bad.mp4":
            raise AudioError("moov atom not found")
        return np.zeros(16_000 * 600, dtype=np.float32)

    warnings: list[str] = []
    scanned = run_scan(
        camera="Street",
        start=T0,
        end=T0 + timedelta(minutes=20),
        output_dir=tmp_path,
        event_config=EventConfig(),
        exporter=lambda **_kwargs: chunks,
        load_audio=load,
        score_audio=lambda _w: [FrameScore(time_seconds=0.0, scores={"Engine": 0.0})],
        progress=lambda *_args: None,
        warn=warnings.append,
    )
    assert scanned.scored == 1
    assert [item.start for item in scanned.failed] == [T0]
    assert "Delete" in scanned.failed[0].error
    assert [chunk.path.name for chunk in scanned.silent] == ["mute.mp4"]
    assert any("no audio track" in warning for warning in warnings)


# Clips


def test_cross_chunk_clip_works_with_a_relative_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("footage").mkdir()
    _video(Path("footage/a.mp4"), 10)
    _video(Path("footage/b.mp4"), 10)
    chunks = [
        FootageChunk(Path("footage/a.mp4"), T0, T0 + timedelta(seconds=10), False),
        FootageChunk(
            Path("footage/b.mp4"), T0 + timedelta(seconds=10), T0 + timedelta(seconds=20), False
        ),
    ]
    run_dir = review_directory(Path("exports"), "O'Brien St", T0)
    media = capture_events(
        [_event(8, 12)], chunks, run_dir=run_dir, padding_seconds=1, write_clips=True
    )
    assert media[0].error is None
    assert media[0].clip is not None and media[0].clip.is_file()


def test_snapshot_past_the_end_of_a_short_file_is_clamped(tmp_path: Path) -> None:
    _video(tmp_path / "short.mp4", 3)
    chunk = FootageChunk(tmp_path / "short.mp4", T0, T0 + timedelta(minutes=10), False)
    media = capture_events([_event(100, 110)], [chunk], run_dir=tmp_path / "run", snapshot_count=2)
    assert media[0].error is None
    assert len(media[0].snapshots) == 2


def test_one_event_failing_does_not_stop_the_others(tmp_path: Path) -> None:
    _video(tmp_path / "a.mp4", 10)
    chunk = FootageChunk(tmp_path / "a.mp4", T0, T0 + timedelta(seconds=10), False)
    media = capture_events(
        [_event(1, 3), _event(50, 60), _event(4, 6)],
        [chunk],
        run_dir=tmp_path / "run",
        write_clips=True,
        padding_seconds=0,
    )
    assert media[0].clip is not None
    assert media[1].clip is None and media[1].error
    assert media[2].clip is not None


def test_capture_preserves_existing_review_files(tmp_path: Path) -> None:
    _video(tmp_path / "a.mp4", 10)
    chunk = FootageChunk(tmp_path / "a.mp4", T0, T0 + timedelta(seconds=10), False)
    run_dir = tmp_path / "run"
    (run_dir / "clips").mkdir(parents=True)
    (run_dir / "clips" / "0009.mp4").write_bytes(b"old")
    (run_dir / "clips" / "keep.txt").write_text("mine", encoding="utf-8")
    capture_events([_event(1, 3)], [chunk], run_dir=run_dir, write_clips=True)
    assert (run_dir / "clips" / "0009.mp4").read_bytes() == b"old"
    assert (run_dir / "clips" / "keep.txt").exists()
    assert (run_dir / "clips" / "0001.mp4").exists()


def test_capture_does_not_overwrite_existing_clip(tmp_path: Path) -> None:
    _video(tmp_path / "a.mp4", 10)
    chunk = FootageChunk(tmp_path / "a.mp4", T0, T0 + timedelta(seconds=10), False)
    run_dir = tmp_path / "run"
    (run_dir / "clips").mkdir(parents=True)
    old = run_dir / "clips" / "0001.mp4"
    old.write_bytes(b"retained")
    media = capture_events([_event(1, 3)], [chunk], run_dir=run_dir, write_clips=True)
    assert old.read_bytes() == b"retained"
    assert media[0].error and "already exists" in media[0].error


def test_failed_clip_removes_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _video(tmp_path / "a.mp4", 10)
    chunk = FootageChunk(tmp_path / "a.mp4", T0, T0 + timedelta(seconds=10), False)
    from protect_audio_search.audio import AudioError

    def fail(_segments: object, dest: Path) -> None:
        dest.write_bytes(b"partial")
        raise AudioError("clip failed")

    monkeypatch.setattr("protect_audio_search.clips._write_clip", fail)
    media = capture_events([_event(1, 3)], [chunk], run_dir=tmp_path / "run", write_clips=True)
    assert media[0].error and "clip failed" in media[0].error
    assert not (tmp_path / "run" / "clips" / "0001.mp4").exists()


def test_partial_snapshot_failure_keeps_successes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _video(tmp_path / "a.mp4", 10)
    chunk = FootageChunk(tmp_path / "a.mp4", T0, T0 + timedelta(seconds=10), False)
    from protect_audio_search import clips
    from protect_audio_search.audio import AudioError

    original = clips._ffmpeg
    calls = 0

    def fail_second(command: list[str], *args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AudioError("snapshot failed")
        original(command, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(clips, "_ffmpeg", fail_second)
    media = capture_events([_event(1, 3)], [chunk], run_dir=tmp_path / "run", snapshot_count=3)
    assert len(media[0].snapshots) == 2
    assert media[0].error and "snapshot 2" in media[0].error


# Confirmation and output


class Flaky:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _images: object) -> ProviderReply:
        self.calls += 1
        if self.calls == 1:
            raise ConfirmError("Anthropic request failed (InternalServerError).")
        return ProviderReply(text='{"vehicle_present": true, "note": "car"}', estimated_usd=0.001)


def test_provider_error_marks_one_event_and_carries_on(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(b"\xff\xd8")
    results, summary = confirm_events([[image], [image]], Flaky(), ConfirmBudget())
    assert "failed" in results[0].note
    assert results[1].vehicle_present is True
    assert summary.failed == 1


def test_call_cap_and_cost_cap_have_their_own_messages(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(b"\xff\xd8")

    class Cheap:
        def complete(self, _images: object) -> ProviderReply:
            return ProviderReply(text="{}", estimated_usd=1.0)

    by_calls, _ = confirm_events([[image]] * 2, Cheap(), ConfirmBudget(max_calls=1, max_cost_usd=9))
    assert "call cap" in by_calls[1].note
    by_cost, _ = confirm_events([[image]] * 2, Cheap(), ConfirmBudget(max_cost_usd=0.5))
    assert "cost cap" in by_cost[1].note


def test_model_text_is_cleaned() -> None:
    parsed = parse_confirmation(
        '{"plate_text": "AB\\u001b[31mC12345678901234567890", "note": "x\\u0007y"}'
    )
    assert parsed.plate_text is not None
    assert "\x1b" not in parsed.plate_text
    assert len(parsed.plate_text) <= 16
    assert "\x07" not in parsed.note


def test_csv_cells_cannot_start_a_formula(tmp_path: Path) -> None:
    from protect_audio_search.confirm import Confirmation

    result = ScanResult(
        camera="=cmd",
        timezone="UTC",
        start=T0,
        end=T0 + timedelta(minutes=10),
        events=(_event(1, 3),),
        confirmations=(Confirmation(True, False, '=HYPERLINK("http://x")', 0.5, "+1"),),
    )
    path = tmp_path / "out.csv"
    write_csv(path, result, ZoneInfo("UTC"))
    row = next(csv.DictReader(path.open(encoding="utf-8")))
    assert row["plate_text"].startswith("'=")
    assert row["confirm_note"] == "'+1"
    assert row["number"] == "1"
