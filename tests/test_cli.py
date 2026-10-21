"""CLI tests with a mocked scan pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jsonschema
import pytest

from protect_audio_search.audio import FfmpegMissingError
from protect_audio_search.cli import main
from protect_audio_search.events import DetectionEvent
from protect_audio_search.footage import Camera

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "scan-result.schema.json"
START = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
END = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)


def _schema() -> dict[str, object]:
    loaded: dict[str, object] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return loaded


def _event() -> DetectionEvent:
    return DetectionEvent(
        camera="Street",
        start=START,
        end=START + timedelta(seconds=5),
        peak_class="Engine",
        peak_score=0.9,
        mean_score=0.8,
        contributing_classes=("Engine", "Motorcycle"),
    )


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert "protect-audio-search" in captured.out
    assert "tyre squeal" in captured.out


def test_scan_help_documents_every_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["scan", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    for flag in (
        "--camera",
        "--from",
        "--to",
        "--tz",
        "--chunk-minutes",
        "--output-dir",
        "--json",
        "--csv",
        "--dry-run",
        "--threshold",
        "--class-threshold",
        "--smooth-frames",
        "--gap",
        "--min-duration",
    ):
        assert flag in text
    assert "docs/scan-result.schema.json" in text
    assert "Australia/Sydney" in text


def test_missing_scan_args_exit_nonzero() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["scan"])
    assert caught.value.code != 0


def test_dry_run_uses_sydney_and_does_not_export(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_prepare() -> None:
        raise AssertionError("dry-run must not check ffmpeg")

    def fail_scan(**_kwargs: object) -> list[DetectionEvent]:
        raise AssertionError("dry-run must not scan")

    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", fail_prepare)
    monkeypatch.setattr("protect_audio_search.cli.run_scan", fail_scan)
    main(
        [
            "scan",
            "--camera",
            "Street",
            "--from",
            "2026-10-20 21:00",
            "--to",
            "2026-10-21 02:00",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "Planned chunks: 30" in out
    assert "chunk 1/30 2026-10-20T21:00:00+11:00 to 2026-10-20T21:10:00+11:00" in out
    assert "chunk 30/30 2026-10-21T01:50:00+11:00 to 2026-10-21T02:00:00+11:00" in out


def test_timezone_override_changes_the_plan(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "scan",
            "--camera",
            "Street",
            "--from",
            "2026-10-20 21:00",
            "--to",
            "2026-10-20 21:10",
            "--tz",
            "Australia/Perth",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "2026-10-20T21:00:00+08:00" in out


def test_zulu_input_displays_in_sydney(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "scan",
            "--camera",
            "Street",
            "--from",
            "2026-10-20T10:00:00Z",
            "--to",
            "2026-10-20T10:10:00Z",
            "--dry-run",
        ]
    )
    assert "2026-10-20T21:00:00+11:00" in capsys.readouterr().out


def test_scan_prints_hits_json_and_csv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def fake_prepare() -> None:
        seen["prepared"] = True

    def fake_scan(**kwargs: object) -> list[DetectionEvent]:
        seen["kwargs"] = kwargs
        progress = kwargs["progress"]
        assert callable(progress)
        progress(1, 30, START, START + timedelta(minutes=10))
        return [_event()]

    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", fake_prepare)
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
            "2026-10-21 02:00",
            "--json",
            str(json_path),
            "--csv",
            str(csv_path),
            "--threshold",
            "0.4",
        ]
    )
    captured = capsys.readouterr()
    assert "2026-10-20T21:00:00+11:00" in captured.out
    assert "Engine" in captured.out
    assert "chunk 1/30 2026-10-20T21:00:00+11:00 to 2026-10-20T21:10:00+11:00" in captured.err
    assert seen["prepared"] is True
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["camera"] == "Street"
    assert kwargs["start"] == START
    assert kwargs["end"] == END
    config = kwargs["event_config"]
    assert config.threshold == pytest.approx(0.4)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, _schema())
    assert payload["schema_version"] == 1
    assert payload["timezone"] == "Australia/Sydney"
    assert payload["start_utc"] == "2026-10-20T10:00:00Z"
    assert payload["end_utc"] == "2026-10-20T15:00:00Z"
    assert payload["events"][0]["start"] == "2026-10-20T21:00:00+11:00"
    assert payload["events"][0]["contributing_classes"] == ["Engine", "Motorcycle"]

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "peak_class" in csv_text
    assert "Engine|Motorcycle" in csv_text
    assert "2026-10-20T21:00:00+11:00" in csv_text


def test_no_hits_exits_zero_and_writes_empty_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr("protect_audio_search.cli.run_scan", lambda **_kwargs: [])
    json_path = tmp_path / "empty.json"
    main(
        [
            "scan",
            "--camera",
            "Street",
            "--from",
            "2026-10-20 21:00",
            "--to",
            "2026-10-20 21:10",
            "--json",
            str(json_path),
        ]
    )
    assert capsys.readouterr().out.strip() == "No events."
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, _schema())
    assert payload["events"] == []


def test_bad_time_and_timezone_exit_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as bad_time:
        main(
            [
                "scan",
                "--camera",
                "Street",
                "--from",
                "yesterday",
                "--to",
                "2026-10-21 02:00",
            ]
        )
    assert bad_time.value.code == 2
    assert "Could not parse time" in capsys.readouterr().err

    with pytest.raises(SystemExit) as bad_zone:
        main(
            [
                "scan",
                "--camera",
                "Street",
                "--from",
                "2026-10-20 21:00",
                "--to",
                "2026-10-21 02:00",
                "--tz",
                "Mars/Olympus",
            ]
        )
    assert bad_zone.value.code == 2
    assert "Unknown timezone" in capsys.readouterr().err


def test_end_before_start_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "scan",
                "--camera",
                "Street",
                "--from",
                "2026-10-21 02:00",
                "--to",
                "2026-10-20 21:00",
                "--dry-run",
            ]
        )
    assert caught.value.code == 2
    assert "end must be after start" in capsys.readouterr().err


def test_missing_ffmpeg_exits_before_scan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_prepare() -> None:
        raise FfmpegMissingError("ffmpeg is not on PATH, so no footage was exported.")

    def fail_scan(**_kwargs: object) -> list[DetectionEvent]:
        raise AssertionError("scan must not start after ffmpeg is missing")

    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", fail_prepare)
    monkeypatch.setattr("protect_audio_search.cli.run_scan", fail_scan)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "scan",
                "--camera",
                "Street",
                "--from",
                "2026-10-20 21:00",
                "--to",
                "2026-10-20 21:10",
            ]
        )
    assert caught.value.code == 1
    assert "no footage was exported" in capsys.readouterr().err


def test_classes_lists_targets_and_context(capsys: pytest.CaptureFixture[str]) -> None:
    main(["classes"])
    out = capsys.readouterr().out
    assert "Accelerating, revving, vroom" in out
    assert "Tire squeal" in out
    assert "Vehicle" in out


def test_cameras_prints_id_and_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "protect_audio_search.cli.list_cameras_sync",
        lambda: [Camera(id="cam-1", name="Street"), Camera(id="cam-2", name="Lane")],
    )
    main(["cameras"])
    assert capsys.readouterr().out == "cam-1\tStreet\ncam-2\tLane\n"


def test_cameras_empty(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("protect_audio_search.cli.list_cameras_sync", lambda: [])
    main(["cameras"])
    assert capsys.readouterr().out.strip() == "No cameras."
