"""CLI checks that run before export, exit codes, and a full CLI run through the mock."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import jsonschema
import pytest

from protect_audio_search import __version__
from protect_audio_search.classifier import ModelCacheError
from protect_audio_search.cli import main
from protect_audio_search.scan import ScanOutput

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tests" / "mock_protect_server.py"
ENGINE = ROOT / "tests" / "fixtures" / "engine_cc0.wav"
SCHEMA = json.loads((ROOT / "docs" / "scan-result.schema.json").read_text(encoding="utf-8"))
BASE = ["scan", "--camera", "Street", "--from", "2026-10-20 21:00", "--to", "2026-10-20 21:10"]


def _no_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**_kwargs: object) -> ScanOutput:
        raise AssertionError("must fail before the scan starts")

    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr("protect_audio_search.cli.run_scan", fail)


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--class-threshold", "Tyre squeal=0.5"], "Tire squeal"),
        (["--class-threshold", "Vehicle=0.1"], "context class"),
        (["--threshold", "nan"], "between 0 and 1"),
        (["--threshold", "1.5"], "between 0 and 1"),
        (["--class-threshold", "Engine=-1"], "between 0 and 1"),
        (["--clip-padding", "inf"], "clip padding"),
        (["--chunk-minutes", "10000000000"], "chunk minutes"),
        (["--gap", "nan"], "gap"),
        (["--confirm", "--snapshots", "1", "--confirm-max-cost", "nan"], "cost cap"),
    ],
)
def test_bad_flags_exit_2_before_any_work(
    flags: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _no_scan(monkeypatch)
    with pytest.raises(SystemExit) as caught:
        main([*BASE, *flags])
    assert caught.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_api_key_fails_before_the_scan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_scan(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as caught:
        main([*BASE, "--snapshots", "1", "--confirm"])
    assert caught.value.code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_json_path_that_is_a_directory_fails_before_the_scan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _no_scan(monkeypatch)
    with pytest.raises(SystemExit) as caught:
        main([*BASE, "--json", str(tmp_path)])
    assert caught.value.code == 2


def test_model_cache_error_exits_1_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def offline(**_kwargs: object) -> ScanOutput:
        raise ModelCacheError("yamnet.tflite is not in the cache")

    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr("protect_audio_search.cli.run_scan", offline)
    with pytest.raises(SystemExit) as caught:
        main(BASE)
    assert caught.value.code == 1
    assert "not in the cache" in capsys.readouterr().err


def test_interrupt_exits_130_and_says_how_to_resume(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(**_kwargs: object) -> ScanOutput:
        raise KeyboardInterrupt

    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr("protect_audio_search.cli.run_scan", interrupted)
    with pytest.raises(SystemExit) as caught:
        main(BASE)
    assert caught.value.code == 130
    assert "resume" in capsys.readouterr().err


def test_no_audio_scored_exits_1_after_writing_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr("protect_audio_search.cli.prepare_scan", lambda: None)
    monkeypatch.setattr(
        "protect_audio_search.cli.run_scan",
        lambda **_kwargs: ScanOutput(events=[], chunks=[], scored=0, planned=1),
    )
    json_path = tmp_path / "out.json"
    with pytest.raises(SystemExit) as caught:
        main([*BASE, "--json", str(json_path)])
    assert caught.value.code == 1
    assert "No audio was scored" in capsys.readouterr().err
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, SCHEMA)
    assert payload["run"]["chunks_scored"] == 0


def _mock_command(tmp_path: Path, **settings: str) -> str:
    config = tmp_path / "mock.json"
    config.write_text(
        json.dumps(
            {
                "MOCK_CALL_LOG": str(tmp_path / "calls.jsonl"),
                "MOCK_EXPORT_DIR": str(tmp_path / "server-cache"),
                "MOCK_MEDIA_WAV": str(ENGINE),
                **settings,
            }
        ),
        encoding="utf-8",
    )
    return shlex.join([sys.executable, str(SERVER), str(config)])


def test_full_cli_scan_through_the_mock_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROTECT_AUDIO_SEARCH_MCP_COMMAND", _mock_command(tmp_path))
    main(
        [
            "scan",
            "--camera",
            "street",
            "--from",
            "2026-10-20T10:00:00Z",
            "--to",
            "2026-10-20T10:20:00Z",
            "--clips",
            "--snapshots",
            "1",
            "--json",
            "out.json",
            "--csv",
            "out.csv",
            "--cleanup-chunks",
        ]
    )
    captured = capsys.readouterr()
    assert "export 1/2" in captured.err
    assert "Scored 2 of 2 exported chunks" in captured.err
    assert "Motorcycle" in captured.out
    payload = json.loads(Path("out.json").read_text(encoding="utf-8"))
    jsonschema.validate(payload, SCHEMA)
    assert payload["run"]["chunks_planned"] == 2
    assert payload["run"]["chunks_scored"] == 2
    assert len(payload["run"]["chunks_short_audio"]) == 2
    first = payload["events"][0]
    assert first["number"] == 1
    assert Path(first["clip"]).is_file()
    assert Path(payload["run"]["review_dir"], "results.json").is_file()
    # The server wrote straight into our chunk path, and cleanup removed it.
    assert not (tmp_path / "server-cache").exists()
    assert not list(Path("exports").rglob("*.mp4.partial"))
    assert not list(Path("exports", "cam-1").glob("*"))
    calls = [json.loads(line) for line in Path("calls.jsonl").read_text().splitlines()]
    assert {call["channel"] for call in calls} == {0}


def test_check_reports_a_camera_that_records_no_audio(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cameras = json.dumps([{"id": "cam-1", "name": "Street", "recordsAudio": False}])
    monkeypatch.setenv(
        "PROTECT_AUDIO_SEARCH_MCP_COMMAND", _mock_command(tmp_path, MOCK_CAMERAS=cameras)
    )
    with pytest.raises(SystemExit) as caught:
        main(["check", "--camera", "Street"])
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert "ok    MCP server started and listed 1 cameras" in captured.out
    assert "does not record audio" in captured.err


def test_check_passes_for_a_camera_with_audio(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("PROTECT_AUDIO_SEARCH_MCP_COMMAND", _mock_command(tmp_path))
    main(["check", "--camera", "Street"])
    out = capsys.readouterr().out
    assert "records audio" in out
    assert "recording mode is always" in out
