"""YAMNet scores for the classes this tool treats as hits and context.

The TFLite model scores one 0.975 second window (15600 samples at 16 kHz) and
returns 521 class scores. Windows hop by 0.48 seconds. Class indices come from
the downloaded CSV. A name that is not in that file is an error.

`WindowScorer` is the backend interface. `TFLiteYAMNet` is the implementation
shipped here. An ONNX backend can implement the same method later.
"""

from __future__ import annotations

import csv
import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from numpy.typing import NDArray

MODEL_URL = "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
MODEL_SHA256 = "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)
CLASS_MAP_SHA256 = "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"

WINDOW_SAMPLES = 15_600
HOP_SAMPLES = 7_680
HOP_SECONDS = HOP_SAMPLES / 16_000
CLASS_COUNT = 521

DEFAULT_TARGET_CLASSES: tuple[str, ...] = (
    "Accelerating, revving, vroom",
    "Skidding",
    "Tire squeal",
    "Race car, auto racing",
    "Engine",
    "Light engine (high frequency)",
    "Medium engine (mid frequency)",
    "Motorcycle",
)
DEFAULT_CONTEXT_CLASSES: tuple[str, ...] = (
    "Vehicle",
    "Motor vehicle (road)",
    "Car",
    "Car passing by",
    "Idling",
    "Engine starting",
    "Traffic noise, roadway noise",
)


class ModelHashError(Exception):
    """A downloaded or cached model file does not match its pinned SHA-256."""


class ModelCacheError(Exception):
    """A model file is missing and downloads are turned off."""


class UnknownClassError(Exception):
    """A configured class name is not in the YAMNet class map."""

    def __init__(self, name: str, suggestions: list[str]) -> None:
        listed = ", ".join(suggestions) if suggestions else "(none)"
        super().__init__(f"Unknown YAMNet class {name!r}. Close matches: {listed}.")
        self.name = name
        self.suggestions = suggestions


@dataclass(frozen=True)
class FrameScore:
    """Scores for the requested classes at one window."""

    time_seconds: float
    scores: dict[str, float]

    @property
    def top_class(self) -> str:
        return max(self.scores, key=self.scores.__getitem__)

    @property
    def top_score(self) -> float:
        return self.scores[self.top_class]


class WindowScorer(Protocol):
    """Score one 16 kHz mono window. Returns one score per YAMNet class."""

    def score_window(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return 521 scores for a window of `WINDOW_SAMPLES`."""


class ClassMap:
    """Display-name lookup over the YAMNet CSV. Indices are never hard-coded."""

    def __init__(self, names: list[str]) -> None:
        if len(names) != CLASS_COUNT:
            raise ValueError(f"YAMNet class map has {len(names)} rows, expected {CLASS_COUNT}")
        self.names = names
        self._index = {name: index for index, name in enumerate(names)}

    def index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError as exc:
            raise UnknownClassError(name, _suggestions(name, self.names)) from exc

    def indices(self, names: list[str] | tuple[str, ...]) -> dict[str, int]:
        return {name: self.index(name) for name in names}


class TFLiteYAMNet:
    """YAMNet classification TFLite model."""

    def __init__(self, model_path: Path) -> None:
        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input = int(self._interpreter.get_input_details()[0]["index"])
        self._output = int(self._interpreter.get_output_details()[0]["index"])

    def score_window(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        if window.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"window must be length {WINDOW_SAMPLES}, got {window.shape}")
        self._interpreter.set_tensor(self._input, np.ascontiguousarray(window, dtype=np.float32))
        self._interpreter.invoke()
        scores = np.asarray(self._interpreter.get_tensor(self._output), dtype=np.float32)
        return scores.reshape(CLASS_COUNT)


def default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "protect-audio-search"


def load_classifier(
    cache_dir: Path | None = None,
    *,
    class_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[TFLiteYAMNet, ClassMap, tuple[str, ...]]:
    """Download the model on first use, check hashes, and resolve class names."""
    model_path, map_path = ensure_assets(cache_dir or default_cache_dir())
    class_map = load_class_map(map_path)
    selected = (
        tuple(class_names)
        if class_names is not None
        else (DEFAULT_TARGET_CLASSES + DEFAULT_CONTEXT_CLASSES)
    )
    class_map.indices(selected)
    return TFLiteYAMNet(model_path), class_map, selected


def classify(
    samples: NDArray[np.float32],
    class_names: tuple[str, ...] | list[str],
    scorer: WindowScorer,
    class_map: ClassMap,
) -> list[FrameScore]:
    """Score `samples` (16 kHz mono, [-1, 1]) at each hop."""
    wanted = class_map.indices(tuple(class_names))
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if waveform.size < WINDOW_SAMPLES:
        waveform = np.pad(waveform, (0, WINDOW_SAMPLES - waveform.size))
    frames: list[FrameScore] = []
    start = 0
    while start + WINDOW_SAMPLES <= waveform.size:
        window = waveform[start : start + WINDOW_SAMPLES]
        raw = scorer.score_window(window)
        if raw.shape != (CLASS_COUNT,):
            raise ValueError(f"backend returned shape {raw.shape}, expected {(CLASS_COUNT,)}")
        frames.append(
            FrameScore(
                time_seconds=start / 16_000,
                scores={name: float(raw[index]) for name, index in wanted.items()},
            )
        )
        start += HOP_SAMPLES
    return frames


def ensure_assets(cache_dir: Path) -> tuple[Path, Path]:
    """Return cached model and class map paths, downloading them on first use."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "yamnet.tflite"
    map_path = cache_dir / "yamnet_class_map.csv"
    _ensure_file(model_path, MODEL_URL, MODEL_SHA256)
    _ensure_file(map_path, CLASS_MAP_URL, CLASS_MAP_SHA256)
    return model_path, map_path


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ModelHashError(
            f"{path.name} SHA-256 is {digest}, expected {expected}. "
            "The file was not used. Delete it from the cache and run again."
        )


def load_class_map(path: Path) -> ClassMap:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["display_name"] for row in rows]
    return ClassMap(names)


def _ensure_file(path: Path, url: str, expected: str) -> None:
    if path.exists():
        verify_sha256(path, expected)
        return
    if os.environ.get("PROTECT_AUDIO_SEARCH_OFFLINE") == "1":
        raise ModelCacheError(
            f"{path.name} is not in the cache at {path.parent}, "
            "and PROTECT_AUDIO_SEARCH_OFFLINE=1 disables downloads."
        )
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "protect-audio-search"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as handle,
        ):
            handle.write(response.read())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    try:
        verify_sha256(temporary, expected)
    except ModelHashError:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)


def _suggestions(name: str, names: list[str]) -> list[str]:
    folded = name.casefold()
    contains = [candidate for candidate in names if folded in candidate.casefold()]
    if contains:
        return contains[:5]
    prefix = [candidate for candidate in names if candidate.casefold().startswith(folded[:4])]
    return prefix[:5]
