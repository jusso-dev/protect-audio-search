# protect-audio-search

On-demand command line tool. It pulls recorded footage from a UniFi Protect console through an existing MCP server, then searches the audio for revving, burnouts and tyre squeal. Hits are timestamped. Each hit can keep a trimmed clip and JPEG snapshots on this machine.

This is not an always-on listener. You choose a camera and a time range, run a scan, and read the result. The package is not published on PyPI. Install it from a clone of this repository.

UniFi Protect's built-in audio detections cover smoke and CO alarms, sirens, car horns, car alarms, glass breaking, barking, speech and baby crying. They do not cover revving, burnouts or tyre squeal. The listening here is [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) (AudioSet, 521 classes) running as TensorFlow Lite. Looking at frames stays with a person, or with the optional vision check in [docs/vision-confirmation.md](docs/vision-confirmation.md).

## Install

You need:

- Python 3.11 or newer
- `ffmpeg` and `ffprobe` on `PATH` (a real scan, clips and snapshots)
- Node.js with `npx` on `PATH` (the default Protect client launches `npx -y @mgcrea/mcp-unifi-protect`)

On macOS, Homebrew can install all three: `brew install python@3.11 ffmpeg node`. On Debian or Ubuntu: `sudo apt-get update && sudo apt-get install -y python3 python3-venv ffmpeg`, plus a Node.js install that provides `npx`.

```bash
git clone https://github.com/jusso-dev/protect-audio-search.git
cd protect-audio-search
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`protect-audio-search` and `python -m protect_audio_search` run the same program.

The first real scan downloads the YAMNet model and class map into `$XDG_CACHE_HOME/protect-audio-search`, or `~/.cache/protect-audio-search` when `XDG_CACHE_HOME` is unset. Both files are checked against pinned SHA-256 hashes. A mismatch stops the run. Model files are not in git.

`--dry-run` and `classes` do not need ffmpeg, Node, credentials, or that download.

## First scan

### 1. See the class names

```bash
protect-audio-search classes
```

This prints the default target names and the context names. It does not download the model.

### 2. Plan the chunks

A time with no timezone is read in `--tz` (default `Australia/Sydney`) and stored as UTC. Printed times use `--tz`. `--to` is exclusive.

```bash
protect-audio-search scan \
  --camera "Street" \
  --from "2026-10-20 21:00" \
  --to "2026-10-21 02:00" \
  --dry-run
```

That range is five hours in Sydney (AEDT, UTC+11 in October): UTC `2026-10-20T10:00:00Z` to `2026-10-20T15:00:00Z`, thirty chunks of ten minutes. `--dry-run` prints the plan and does not export or classify. Use `--tz Australia/Perth` when the clock you mean is Perth.

An aware timestamp also works: `--from 2026-10-20T10:00:00Z`.

### 3. Point the MCP server at your console

This program does not read a UniFi password. It starts [mgcrea/mcp-unifi-protect](https://github.com/mgcrea/mcp-unifi-protect) and passes through the current environment. Set variables in that environment. Do not commit them, and do not paste real values into this repository.

Local console (the usual case), names only:

- `UNIFI_PROTECT_HOST`
- `UNIFI_PROTECT_USERNAME`
- `UNIFI_PROTECT_PASSWORD`

Cloud mode, names only:

- `UNIFI_PROTECT_API_KEY`
- `UNIFI_PROTECT_CONSOLE_ID`

The full table, TLS pinning, and account setup live in that server's README. Use a dedicated local view-only account. Leave `UNIFI_PROTECT_ALLOW_WRITES` unset. This tool only calls `unifi_protect_list_cameras` and `unifi_protect_export_video`. If the account uses a one-time code, follow the MCP server's own login notes. This CLI has no prompt for a second factor.

```bash
protect-audio-search cameras
```

Each line is `id`, then a tab, then the name. An empty console prints `No cameras.`

### 4. Run a scan

```bash
protect-audio-search scan \
  --camera "Street" \
  --from "2026-10-20 21:00" \
  --to "2026-10-21 02:00" \
  --json results.json \
  --csv results.csv
```

`--camera` is a Protect name (case-insensitive, must be unique) or a camera id. The label stored on each event is the string you passed.

Exit status is 0 when the scan finishes, including when nothing matched. Stdout then says `No events.` Progress is on stderr, in `--tz`. A bad argument exits 2. A failure talking to Protect, ffmpeg, the model cache, or vision confirmation exits 1. Those messages are redacted, but the child MCP process can still write its own errors to stderr.

Footage chunks land in `--output-dir` (default `exports`):

```text
exports/<camera-id>/20261020T100000Z_20261020T101000Z.mp4
```

An existing non-empty chunk file is skipped, so a rerun can resume.

Add review files when you want to look as well as listen:

```bash
protect-audio-search scan \
  --camera "Street" \
  --from "2026-10-20 21:00" \
  --to "2026-10-20 21:30" \
  --clips \
  --snapshots 3
```

For a Sydney start of `2026-10-20 21:00`, clips and snapshots go under:

```text
exports/20261020T100000Z-Street/clips/0001.mp4
exports/20261020T100000Z-Street/snapshots/0001-01.jpg
```

The folder stamp is the range start in UTC. `--clip-padding` defaults to 5 seconds around the event. Snapshots are spaced across the event itself, not the padding. `--cleanup-chunks` deletes the exported footage files after those review files are written, and leaves the review folder.

## Configuration

Variables this package reads:

| Variable | When |
|---|---|
| `ANTHROPIC_API_KEY` | Only with `--confirm`. Never printed. Missing key exits 2 and names the variable. |
| `ANTHROPIC_MODEL` | Optional. Default `claude-haiku-5-5`. |
| `XDG_CACHE_HOME` | Optional. Model cache directory. |
| `PROTECT_AUDIO_SEARCH_OFFLINE` | Set to `1` to refuse a download. A file already in the cache is still hash-checked. |

UniFi variables are listed under the first-scan step above. They belong to the MCP server.

## Tuning

Defaults: score threshold `0.3`, smoothing off (`--smooth-frames 1`), gap `2` seconds, minimum duration `1.5` seconds. YAMNet windows are `0.975` seconds and hop every `0.48` seconds, so one loud frame is dropped and three hops (about `1.9` seconds) can be kept.

```bash
protect-audio-search scan \
  --camera "Street" \
  --from "2026-10-20 21:00" \
  --to "2026-10-20 22:00" \
  --threshold 0.45 \
  --class-threshold "Tire squeal=0.5" \
  --class-threshold "Engine=0.2" \
  --smooth-frames 3 \
  --gap 2 \
  --min-duration 1.5
```

- Run a short range you can check by ear before a long export.
- Too many hits (motorbikes, trucks, leaf blowers, loud music): raise `--threshold`, or raise one class with `--class-threshold NAME=SCORE`. Raising a threshold does not add events.
- Missed sounds you can hear: lower that class. If the sound is shorter than `--min-duration`, lower that too.
- `--smooth-frames` is a trailing moving average. `1` is off. A longer average hides one-frame spikes and can also hide a short squeal.
- `--gap` joins frames that sit at most that many seconds apart. A larger gap means fewer, longer events. Chunks join the same way, by absolute time.
- Tune per camera. A camera on a road is not a camera in a driveway.
- The name in `--class-threshold` must be the AudioSet display name. `Tire squeal` uses the American spelling from AudioSet. An unknown name exits with a few close matches.
- Context classes such as `Vehicle` often score above `0.8` while `Engine` on the same window is much lower. A scan does not treat the global top class as the hit. The peak on an event is the highest smoothed score among target classes that passed their own threshold.

There is no flag to turn a context class into a hit.

## What a scan does

1. Check that `ffmpeg` is installed (skipped for `--dry-run`).
2. Start `npx -y @mgcrea/mcp-unifi-protect` over stdio. The launch command is not a CLI flag.
3. Resolve `--camera`, then export the range in chunks (default 10 minutes, last chunk shorter). Transient export errors retry. A permanent error fails that chunk once.
4. Extract 16 kHz mono audio. A chunk with no audio stream is skipped, with a warning on stderr.
5. Score each window with the TFLite YAMNet model. `WindowScorer` is a small interface so another backend could be added later. No ONNX backend ships.
6. Merge target-class frames into events.
7. Optionally write clips and snapshots, optionally send snapshots to Anthropic, then optionally delete the footage chunks.

JSON (`--json`) follows [docs/scan-result.schema.json](docs/scan-result.schema.json) (`schema_version` 1). The file has `start_utc` and `end_utc`. Each event's `start` and `end` are in `--tz` with a numeric offset. `confirmation` is `null` when `--confirm` is off. CSV (`--csv`) always includes the confirmation columns, empty when confirmation is off. The terminal table is plain aligned text.

## Classes

Default targets, exact AudioSet display names:

| YAMNet index | AudioSet id | Display name |
|---|---|---|
| 347 | /m/07q2z82 | Accelerating, revving, vroom |
| 306 | /m/07rknqz | Skidding |
| 307 | /m/0h9mv | Tire squeal |
| 309 | /m/0ltv | Race car, auto racing |
| 337 | /m/02mk9 | Engine |
| 338 | /t/dd00065 | Light engine (high frequency) |
| 342 | /t/dd00066 | Medium engine (mid frequency) |
| 320 | /m/04_sv | Motorcycle |

Context names, printed by `classes` and not used as hits: `Vehicle` (294), `Motor vehicle (road)` (300), `Car` (301), `Car passing by` (308), `Idling` (346), `Engine starting` (345), `Traffic noise, roadway noise` (321).

## Vision confirmation

`--confirm` is off unless you set it. It requires `--snapshots` of at least 1. Snapshot JPEGs are sent to Anthropic and leave this machine. Caps default to 3 images per event, 20 calls, and USD 0.50 estimated cost. Details, the cost rates, and the skip rules are in [docs/vision-confirmation.md](docs/vision-confirmation.md).

## Limitations

- **It only hears the camera.** Many cameras have no microphone, or the microphone is disabled. Distance, wind, rain and walls all hurt accuracy.
- **YAMNet is a general model.** It was not trained to find hoon driving. Expect false positives and missed events. The quantised model often ranks `Vehicle` above `Engine` even on revving. Thresholds need tuning per camera.
- **Exports are slow and large.** Hours of footage take time and disk. A five hour range at the default chunk size is thirty downloads.
- **Private API.** The MCP server wraps Protect's private API, which can change between Protect releases.
- **Plates are hints.** Night snapshots and the vision check are not evidence. A plate string can be wrong.
- **TFLite only.** ONNX is not implemented.
- **One-time download.** A real scan needs network access the first time, unless the cache is already warm. `PROTECT_AUDIO_SEARCH_OFFLINE=1` with an empty cache fails instead of downloading.
- **No upload of review files**, except the snapshots you explicitly send with `--confirm`.

## Privacy and legal

This is not legal advice, and it is not a complete list of the law. Recording audio, and keeping images of people or number plates, is regulated. The rules differ between Australian states and territories, and they differ again for a workplace or a public place. Read the current Act for the place the camera records before you enable a microphone or keep clips.

The Commonwealth [Surveillance Devices Act 2004](https://www.legislation.gov.au/C2004A01387) is mainly about law-enforcement warrants. It is not a code of practice for a private camera.

State and territory laws include:

- New South Wales: [Surveillance Devices Act 2007](https://legislation.nsw.gov.au/view/html/inforce/current/act-2007-064) and the Workplace Surveillance Act 2005. Register: [legislation.nsw.gov.au](https://legislation.nsw.gov.au/).
- Victoria: Surveillance Devices Act 1999. Register: [legislation.vic.gov.au](https://www.legislation.vic.gov.au/).
- Queensland: Invasion of Privacy Act 1971. Register: [legislation.qld.gov.au](https://www.legislation.qld.gov.au/).
- Western Australia: Surveillance Devices Act 1998. Register: [legislation.wa.gov.au](https://www.legislation.wa.gov.au/).
- South Australia: Surveillance Devices Act 2016. Register: [legislation.sa.gov.au](https://www.legislation.sa.gov.au/).
- Tasmania: Listening Devices Act 1991. Register: [legislation.tas.gov.au](https://www.legislation.tas.gov.au/).
- Australian Capital Territory: Listening Devices Act 1992. Register: [legislation.act.gov.au](https://www.legislation.act.gov.au/).
- Northern Territory: Surveillance Devices Act 2007. Register: [legislation.nt.gov.au](https://legislation.nt.gov.au/).

The Office of the Australian Information Commissioner publishes privacy guidance at [oaic.gov.au/privacy](https://www.oaic.gov.au/privacy).

Practical limits this tool can keep:

- Footage, clips and snapshots stay on this machine unless you turn on `--confirm`.
- `--confirm` sends images to Anthropic. Read [docs/vision-confirmation.md](docs/vision-confirmation.md) and [anthropic.com](https://www.anthropic.com/) before you do that.
- Delete footage when you no longer need it. `--cleanup-chunks` removes exported chunks and leaves the review folder. It does not decide your retention period.
- Prefer a view-only Protect account used only for this tool.
- Do not commit `.env`, footage, or model files.

## Security and contributing

[SECURITY.md](SECURITY.md) covers reporting, secrets, and what the process is allowed to call. [CONTRIBUTING.md](CONTRIBUTING.md) covers the local checks.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
