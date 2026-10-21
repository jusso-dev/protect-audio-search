"""YAMNet class scores, class-map lookup, and model hash checks."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from protect_audio_search.classifier import (
    CLASS_COUNT,
    DEFAULT_TARGET_CLASSES,
    HOP_SAMPLES,
    WINDOW_SAMPLES,
    ClassMap,
    FrameScore,
    ModelCacheError,
    ModelHashError,
    TFLiteYAMNet,
    UnknownClassError,
    classify,
    default_cache_dir,
    ensure_assets,
    load_classifier,
    verify_sha256,
)

FIXTURE = Path(__file__).parent / "fixtures" / "engine_cc0.wav"


class _FakeScorer:
    def score_window(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        scores = np.zeros(CLASS_COUNT, dtype=np.float32)
        scores[0] = 0.25
        scores[1] = float(window[0])
        return scores


def _read_wav(path: Path) -> NDArray[np.float32]:
    with wave.open(str(path)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16_000
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767


def test_frame_times_follow_the_hop() -> None:
    class_map = ClassMap([f"class-{index}" for index in range(CLASS_COUNT)])
    samples = np.zeros(WINDOW_SAMPLES + HOP_SAMPLES, dtype=np.float32)
    samples[0] = 0.5
    frames = classify(samples, ["class-0", "class-1"], _FakeScorer(), class_map)
    assert [frame.time_seconds for frame in frames] == [0.0, 0.48]
    assert frames[0].scores["class-1"] == pytest.approx(0.5)
    assert frames[1].scores["class-1"] == pytest.approx(0.0)


def test_unknown_class_lists_close_matches() -> None:
    padding = [f"pad-{index}" for index in range(CLASS_COUNT - len(DEFAULT_TARGET_CLASSES))]
    class_map = ClassMap([*DEFAULT_TARGET_CLASSES, *padding])
    with pytest.raises(UnknownClassError, match="Accelerating, revving, vroom") as caught:
        class_map.index("Revving")
    assert caught.value.name == "Revving"


def test_hash_mismatch_aborts(tmp_path: Path) -> None:
    path = tmp_path / "yamnet.tflite"
    path.write_bytes(b"this is not the model")
    with pytest.raises(ModelHashError, match="SHA-256"):
        verify_sha256(path, "0" * 64)
    assert path.exists()


@pytest.fixture(scope="module")
def loaded() -> tuple[TFLiteYAMNet, ClassMap, tuple[str, ...]]:
    """Use the shared cache so a warm cache needs no download."""
    return load_classifier(default_cache_dir())


def test_fixture_top_class_is_a_target(
    loaded: tuple[TFLiteYAMNet, ClassMap, tuple[str, ...]],
) -> None:
    backend, class_map, _selected = loaded
    samples = _read_wav(FIXTURE)
    frames = classify(samples, DEFAULT_TARGET_CLASSES, backend, class_map)
    assert len(frames) == 1
    frame: FrameScore = frames[0]
    assert frame.time_seconds == 0.0
    assert frame.top_class in DEFAULT_TARGET_CLASSES
    assert frame.top_class == "Motorcycle"
    assert frame.top_score > 0.4


def test_cached_hash_mismatch_aborts_before_use(
    loaded: tuple[TFLiteYAMNet, ClassMap, tuple[str, ...]],
    tmp_path: Path,
) -> None:
    del loaded
    cache = tmp_path / "cache"
    cache.mkdir()
    bad = cache / "yamnet.tflite"
    bad.write_bytes(b"corrupt")
    with pytest.raises(ModelHashError, match="was not used"):
        load_classifier(cache)
    assert bad.read_bytes() == b"corrupt"


def test_offline_mode_does_not_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROTECT_AUDIO_SEARCH_OFFLINE", "1")

    def refuse_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline mode must not download")

    monkeypatch.setattr("urllib.request.urlopen", refuse_network)
    with pytest.raises(ModelCacheError, match="disables downloads"):
        ensure_assets(tmp_path)


def test_class_map_rows_match_model(
    loaded: tuple[TFLiteYAMNet, ClassMap, tuple[str, ...]],
) -> None:
    _backend, class_map, _selected = loaded
    assert class_map.index("Engine") != class_map.index("Motorcycle")
    assert class_map.names[class_map.index("Tire squeal")] == "Tire squeal"
