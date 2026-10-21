"""Print how fast YAMNet scores 10 seconds of 16 kHz audio.

Run from the repo root after `pip install -e .`. The model is downloaded on
first use into the cache directory.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

from protect_audio_search.classifier import (
    DEFAULT_TARGET_CLASSES,
    classify,
    load_classifier,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "engine_cc0.wav"
SECONDS = 10


def _fixture_samples() -> np.ndarray:
    with wave.open(str(FIXTURE)) as handle:
        frames = handle.readframes(handle.getnframes())
    window = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767
    repeats = int(np.ceil(SECONDS * 16_000 / window.size))
    return np.tile(window, repeats)[: SECONDS * 16_000]


def main() -> None:
    samples = _fixture_samples()
    backend, class_map, _names = load_classifier()
    started = time.perf_counter()
    frames = classify(samples, DEFAULT_TARGET_CLASSES, backend, class_map)
    elapsed = time.perf_counter() - started
    factor = SECONDS / elapsed
    print(f"samples: {samples.size} ({SECONDS:.2f} s)")
    print(f"frames: {len(frames)}")
    print(f"elapsed: {elapsed:.3f} s")
    print(f"realtime factor: {factor:.1f}x")
    if factor < 20:
        raise SystemExit(f"slower than 20x realtime ({factor:.1f}x)")


if __name__ == "__main__":
    main()
