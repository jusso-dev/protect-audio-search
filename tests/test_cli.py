"""Smoke test for the scaffolded command."""

import pytest

from protect_audio_search.cli import main


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert "protect-audio-search" in captured.out
    assert "tyre squeal" in captured.out
