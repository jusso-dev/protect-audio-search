"""Vision confirmation with a stubbed provider. No live LLM calls."""

from __future__ import annotations

import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jsonschema
import pytest

from protect_audio_search.cli import main
from protect_audio_search.clips import EventMedia
from protect_audio_search.confirm import (
    ConfirmBudget,
    ProviderReply,
    confirm_events,
    estimate_cost_usd,
    parse_confirmation,
)
from protect_audio_search.events import DetectionEvent
from protect_audio_search.scan import ScanOutput

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "scan-result.schema.json"
SECRET = "super-secret-key-value"
GOOD = json.dumps(
    {
        "vehicle_present": True,
        "smoke_present": False,
        "plate_text": "ABC123",
        "plate_confidence": 0.42,
        "note": "Dark car, no smoke.",
    }
)


class Stub:
    def __init__(self, replies: list[str], cost: float = 0.01) -> None:
        self._replies = list(replies)
        self._cost = cost
        self.batches: list[int] = []

    def complete(self, images: list[bytes]) -> ProviderReply:
        self.batches.append(len(images))
        return ProviderReply(text=self._replies.pop(0), estimated_usd=self._cost)


def _event() -> DetectionEvent:
    start = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    return DetectionEvent(
        camera="Street",
        start=start,
        end=start + timedelta(seconds=5),
        peak_class="Engine",
        peak_score=0.9,
        mean_score=0.8,
        contributing_classes=("Engine",),
    )


def _jpeg(path: Path) -> Path:
    path.write_bytes(b"\xff\xd8\xff\xd9")
    return path


def test_estimate_uses_haiku_short_prompt_rates() -> None:
    assert estimate_cost_usd(1_000_000, 1_000_000) == pytest.approx(0.60)


def test_malformed_reply_is_a_hint() -> None:
    parsed = parse_confirmation("not json at all")
    assert parsed.vehicle_present is None
    assert "could not be read" in parsed.note
    parsed_bool = parse_confirmation('{"vehicle_present": "yes", "note": "hmm"}')
    assert parsed_bool.vehicle_present is None
    assert parsed_bool.note == "hmm"


def test_image_cap_and_cost_cap(tmp_path: Path) -> None:
    paths = [_jpeg(tmp_path / f"{index}.jpg") for index in range(5)]
    stub = Stub([GOOD, GOOD], cost=0.2)
    results, summary = confirm_events(
        [paths, paths],
        stub,
        ConfirmBudget(max_images_per_event=3, max_calls=20, max_cost_usd=0.2),
    )
    assert stub.batches == [3]
    assert summary.calls == 1
    assert summary.images == 3
    assert summary.skipped == 1
    assert results[0].plate_text == "ABC123"
    assert "cost cap" in results[1].note


def test_confirm_off_does_not_build_a_provider_or_open_a_socket(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def refuse_provider() -> None:
        raise AssertionError("confirm is off, so no provider")

    def refuse_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("confirm is off, so no network")

    monkeypatch.setattr("protect_audio_search.cli.build_anthropic_provider", refuse_provider)
    monkeypatch.setattr(socket, "create_connection", refuse_socket)
    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr(
        "protect_audio_search.cli.run_scan",
        lambda **_kwargs: ScanOutput(events=[_event()], chunks=[]),
    )
    json_path = tmp_path / "out.json"
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
    assert "leave this machine" not in capsys.readouterr().err
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["events"][0]["confirmation"] is None


def test_confirm_puts_hint_in_the_table_and_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    jpeg = _jpeg(tmp_path / "frame.jpg")
    stub = Stub([GOOD])

    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr(
        "protect_audio_search.cli.run_scan",
        lambda **_kwargs: ScanOutput(events=[_event()], chunks=[]),
    )
    monkeypatch.setattr(
        "protect_audio_search.cli.capture_events",
        lambda *_args, **_kwargs: [EventMedia(clip=None, snapshots=(jpeg,))],
    )
    monkeypatch.setattr("protect_audio_search.cli.build_anthropic_provider", lambda: stub)
    json_path = tmp_path / "out.json"
    main(
        [
            "scan",
            "--camera",
            "Street",
            "--from",
            "2026-10-20 21:00",
            "--to",
            "2026-10-20 21:10",
            "--snapshots",
            "1",
            "--confirm",
            "--json",
            str(json_path),
        ]
    )
    captured = capsys.readouterr()
    assert "leave this machine" in captured.err
    assert "Vision calls: 1" in captured.err
    assert "ABC123" in captured.out
    assert "yes" in captured.out
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    hint = payload["events"][0]["confirmation"]
    assert hint["vehicle_present"] is True
    assert hint["smoke_present"] is False
    assert hint["plate_text"] == "ABC123"
    assert hint["plate_confidence"] == pytest.approx(0.42)


def test_malformed_provider_reply_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jpeg = _jpeg(tmp_path / "frame.jpg")
    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr(
        "protect_audio_search.cli.run_scan",
        lambda **_kwargs: ScanOutput(events=[_event()], chunks=[]),
    )
    monkeypatch.setattr(
        "protect_audio_search.cli.capture_events",
        lambda *_args, **_kwargs: [EventMedia(clip=None, snapshots=(jpeg,))],
    )
    monkeypatch.setattr(
        "protect_audio_search.cli.build_anthropic_provider",
        lambda: Stub(["nope"]),
    )
    json_path = tmp_path / "out.json"
    main(
        [
            "scan",
            "--camera",
            "Street",
            "--from",
            "2026-10-20 21:00",
            "--to",
            "2026-10-20 21:10",
            "--snapshots",
            "1",
            "--confirm",
            "--json",
            str(json_path),
        ]
    )
    hint = json.loads(json_path.read_text(encoding="utf-8"))["events"][0]["confirmation"]
    assert "could not be read" in hint["note"]


def test_missing_key_exits_and_a_set_key_is_not_printed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr(
        "protect_audio_search.cli.run_scan",
        lambda **_kwargs: ScanOutput(events=[_event()], chunks=[]),
    )
    monkeypatch.setattr(
        "protect_audio_search.cli.capture_events",
        lambda *_args, **_kwargs: [EventMedia(clip=None, snapshots=(_jpeg(tmp_path / "a.jpg"),))],
    )
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
                "--snapshots",
                "1",
                "--confirm",
            ]
        )
    assert caught.value.code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setattr(
        "protect_audio_search.confirm.AnthropicVision.complete",
        lambda self, images: ProviderReply(text=GOOD, estimated_usd=0.0),
    )
    main(
        [
            "scan",
            "--camera",
            "Street",
            "--from",
            "2026-10-20 21:00",
            "--to",
            "2026-10-20 21:10",
            "--snapshots",
            "1",
            "--confirm",
        ]
    )
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err
