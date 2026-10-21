"""Clip and snapshot tests against short generated videos."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jsonschema
import pytest

from protect_audio_search.audio import media_duration_seconds
from protect_audio_search.cli import main
from protect_audio_search.clips import capture_events, delete_chunks, review_directory
from protect_audio_search.events import DetectionEvent
from protect_audio_search.footage import FootageChunk
from protect_audio_search.scan import ScanOutput

T0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "scan-result.schema.json"


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", *args], check=True, capture_output=True)


def _chunk_video(path: Path, seconds: int) -> None:
    _ffmpeg(
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
    )


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


def _chunks(directory: Path) -> list[FootageChunk]:
    directory.mkdir(parents=True, exist_ok=True)
    first = directory / "a.mp4"
    second = directory / "b.mp4"
    _chunk_video(first, 10)
    _chunk_video(second, 10)
    return [
        FootageChunk(path=first, start=T0, end=T0 + timedelta(seconds=10), skipped=False),
        FootageChunk(
            path=second,
            start=T0 + timedelta(seconds=10),
            end=T0 + timedelta(seconds=20),
            skipped=False,
        ),
    ]


def test_cross_chunk_clip_covers_padding_and_snapshots(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path)
    run_dir = review_directory(tmp_path / "out", "Street", T0)
    media = capture_events(
        [_event(8, 12)],
        chunks,
        run_dir=run_dir,
        padding_seconds=2,
        write_clips=True,
        snapshot_count=3,
    )
    assert len(media) == 1
    clip = media[0].clip
    assert clip is not None
    assert clip.is_file()
    # 8 s to 12 s plus 2 s each side is 8 s, and it must include both chunks.
    duration = media_duration_seconds(clip)
    assert duration == pytest.approx(8, abs=0.5)
    assert len(media[0].snapshots) == 3
    for path in media[0].snapshots:
        assert path.is_file()
        assert path.read_bytes()[:2] == b"\xff\xd8"
    assert delete_chunks(chunks) == 2
    assert not chunks[0].path.exists()
    assert clip.is_file()


def test_padding_is_clamped_to_available_footage(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path)
    media = capture_events(
        [_event(0.5, 1.5)],
        chunks,
        run_dir=tmp_path / "run",
        padding_seconds=5,
        write_clips=True,
        snapshot_count=1,
    )
    clip = media[0].clip
    assert clip is not None
    # 5 s of leading padding is clamped to the start of the first chunk.
    assert media_duration_seconds(clip) == pytest.approx(6.5, abs=0.5)
    assert len(media[0].snapshots) == 1


def test_cli_puts_clip_and_snapshot_paths_in_json_and_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunks = _chunks(tmp_path / "footage")
    event = _event(8, 12)

    def fake_scan(**_kwargs: object) -> ScanOutput:
        return ScanOutput(events=[event], chunks=chunks)

    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr("protect_audio_search.cli.run_scan", fake_scan)
    json_path = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"
    main(
        [
            "scan",
            "--camera",
            "Street",
            "--from",
            "2026-10-20 21:00",
            "--to",
            "2026-10-20 21:10",
            "--output-dir",
            str(tmp_path / "exports"),
            "--clips",
            "--clip-padding",
            "2",
            "--snapshots",
            "2",
            "--cleanup-chunks",
            "--json",
            str(json_path),
            "--csv",
            str(csv_path),
        ]
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    saved = payload["events"][0]
    assert saved["clip"]
    assert len(saved["snapshots"]) == 2
    assert Path(saved["clip"]).is_file()
    assert all(Path(path).is_file() for path in saved["snapshots"])
    assert not chunks[0].path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "clip" in csv_text.splitlines()[0]
    assert Path(saved["clip"]).name in csv_text
    assert saved["snapshots"][0] in csv_text or Path(saved["snapshots"][0]).name in csv_text
