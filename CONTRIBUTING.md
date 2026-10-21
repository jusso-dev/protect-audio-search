# Contributing

## Setup

Python 3.11 or newer, and `ffmpeg` on `PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Dev extras are Ruff, Mypy, pytest, and jsonschema.

## Checks

```bash
ruff check .
ruff format --check .
mypy
pytest
```

Ruff line length is 100. Mypy is strict for the `protect_audio_search` package only.

GitHub Actions runs those commands on Ubuntu for Python 3.11 and 3.12. The workflow installs ffmpeg, restores a YAMNet cache, warms that cache, then runs pytest with `PROTECT_AUDIO_SEARCH_OFFLINE=1`.

On a developer machine, pytest downloads the model if the cache is cold. After a warm cache:

```bash
PROTECT_AUDIO_SEARCH_OFFLINE=1 pytest
```

`PROTECT_AUDIO_SEARCH_OFFLINE=1` with an empty cache raises `ModelCacheError` and does not download.

## Tests worth knowing

`tests/mock_protect_server.py` stands in for the Protect MCP server. Footage tests use its tiny fake MP4. `tests/test_e2e.py` sets `MOCK_MEDIA_WAV` so the mock muxes a real short file, then runs the real client, ffmpeg, YAMNet, and event detector.

`tests/fixtures/engine_cc0.wav` is a CC0 clip. The expected hit is `Motorcycle`. Silence in the end-to-end test is generated digital zero and must produce no events. The squeal fixture is a generated sine sweep, not a recording, and the test does not require it to be `Tire squeal`.

## House style

- Australian English in docs and in messages this tool prints.
- No em dashes.
- Keep AudioSet display names exact. `Tire squeal` stays in the American spelling from the class map.
- Credentials come from environment variables only. Never log them.
- Do not commit model weights, footage, or secrets.
- Do not add a feature outside the issue you are fixing.
- The shipped classifier backend is TFLite. Do not document ONNX as available.
