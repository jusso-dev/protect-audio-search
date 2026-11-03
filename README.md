# protect-audio-search

On-demand command line tool. It pulls recorded footage from a UniFi Protect console through an existing MCP server, then searches the audio for revving, burnouts and tyre squeal. Hits are timestamped. Each hit can keep a trimmed clip and JPEG snapshots on this machine.

This is not an always-on listener. You choose a camera and a time range, run a scan, and read the result. The package is not published on PyPI. Install it from a clone of this repository.

UniFi Protect's built-in audio detections cover smoke and CO alarms, sirens, car horns, car alarms, glass breaking, barking, speech and baby crying. They do not cover revving, burnouts or tyre squeal. The listening here is [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) (AudioSet, 521 classes) running as TensorFlow Lite. Looking at frames stays with a person, or with the optional vision check in [docs/vision-confirmation.md](docs/vision-confirmation.md).

## Install

You need:

- Python 3.11 or newer
- `ffmpeg` and `ffprobe` on `PATH` (a real scan, clips and snapshots)
- Node.js with `npx` on `PATH` (the default Protect client launches `npx -y @mgcrea/mcp-unifi-protect@1.1.0`, pinned)

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

`--dry-run` and `classes` do not need ffmpeg, Node, credentials, or that download. `protect-audio-search --version` prints the version.

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

This program does not read a UniFi password. It starts [mgcrea/mcp-unifi-protect](https://github.com/mgcrea/mcp-unifi-protect) and passes it an allowlisted copy of the current environment: `UNIFI_PROTECT_*`, `PATH`, `HOME`, locale, proxy, certificate, and Node or npm settings. Other variables, including `ANTHROPIC_API_KEY`, are not passed. Set variables in that environment. Do not commit them, and do not paste real values into this repository.

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

Before a long scan, check the whole chain without exporting anything:

```bash
protect-audio-search check --camera "Street"
```

It checks ffmpeg, downloads and hash-checks the model, starts the MCP server, logs in, and asks Protect whether the camera currently keeps audio in its recordings. If audio is off, it prints a note; older recordings may still have sound. If the server lists only its auth tool, the message names the variables to set.

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

Exit status is 0 when the scan finishes without failed chunks, including when nothing matched. Stdout then says `No events.` If some audio was scored but a chunk failed or was not attempted, the scan writes partial results and exits 3. If no chunk had audio to score, it writes its results and exits 1, because an empty result would otherwise read as "nothing happened". A bad argument exits 2, before anything is exported. A fatal failure talking to Protect, ffmpeg, the model cache, or a file write exits 1. Ctrl-C exits 130 and keeps the exported chunks. Error messages are redacted, but the child MCP process can still write its own errors to stderr.

Progress is on stderr, in `--tz`. `export 3/30` is printed before each download, `cached 3/30` for a chunk already on disk, and `chunk 3/30` when that chunk is scored. Each chunk is scored as soon as it lands. At the end a line says how many chunks were scored, how many had no audio, and how many failed.

A chunk that cannot be exported (no recording in that window, say) or decoded is reported and skipped, and the scan carries on. Authentication errors, second-factor requests and client timeouts stop further downloads; cached chunks are still scored, and unattempted ranges are listed as failed. The JSON `run` block lists every skipped chunk. Run the same command again to retry them.

Footage chunks land in `--output-dir` (default `exports`):

```text
exports/<camera-id>/channel-0/20261020T100000Z_20261020T101000Z.mp4
```

An existing non-empty chunk file for the same camera, channel and range is skipped, so a rerun can resume. Each export has a unique partial path through `savePath`, and the file is renamed into place when complete, so no second copy is left in the server's own cache (`~/.cache/unifi-protect`). A `.json` file next to each chunk keeps the export's real start time for a resumed run. A timed-out server may still write a partial file later; check the export folder before removing it.

The Protect export starts on the keyframe before the requested time, up to a second or so early. Event times use the export's reported `startsAt`, and the lead-in is not scored twice.

The MCP server refuses a download over `UNIFI_PROTECT_MAX_DOWNLOAD_BYTES` (200 MB by default). Full-quality footage can pass that in ten minutes. A chunk over the limit is split in half and retried, down to one-minute pieces. To keep downloads smaller from the start, pass `--channel 1` or `--channel 2` (lower bitrate, same audio, softer snapshots) or a shorter `--chunk-minutes`.

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
exports/20261020T100000Z-Street/run-<unique>/clips/0001.mp4
exports/20261020T100000Z-Street/run-<unique>/snapshots/0001-01.jpg
```

The folder stamp is the range start in UTC. File numbers match the `#` column in the table and `number` in the JSON. The run folder also gets a `results.json`, and its full path is printed at the end. Each run gets a new folder, preserving earlier review files. `--clip-padding` defaults to 5 seconds around the event. Snapshots are spaced across the event itself, not the padding. A clip or snapshot that cannot be written is reported on that event (`media_error`) and the others carry on.

`--cleanup-chunks` deletes exported footage after results and review files are written when processing is complete. Failed, silent or short chunks and review-media errors cause the footage to be retained for recovery. An interrupted run also keeps its chunks.

Files the scan creates (chunks, clips, snapshots, results) are readable only by your user.

For the first long run, follow [docs/real-scan-checklist.md](docs/real-scan-checklist.md).

## Configuration

Variables this package reads:

| Variable | When |
|---|---|
| `ANTHROPIC_API_KEY` | Only with `--confirm`. Never printed. Missing key exits 2 and names the variable. |
| `ANTHROPIC_MODEL` | Optional. Default `claude-haiku-5-5`. |
| `XDG_CACHE_HOME` | Optional. Model cache directory. |
| `PROTECT_AUDIO_SEARCH_OFFLINE` | Set to `1` to refuse a download. A file already in the cache is still hash-checked. |
| `PROTECT_AUDIO_SEARCH_MCP_COMMAND` | Optional. Replaces `npx -y @mgcrea/mcp-unifi-protect@1.1.0`, for example with a locally installed server. Split like a shell command line. |

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
- `--gap` joins frames whose window starts sit at most that many seconds apart. A larger gap means fewer, longer events. Chunks join the same way, by absolute time. A gap below the `0.48` second hop splits every frame into its own event.
- Tune per camera. A camera on a road is not a camera in a driveway.
- The name in `--class-threshold` must be one of the target display names below. `Tire squeal` uses the American spelling from AudioSet. Any other name exits 2 with close matches. A context class name exits 2 too, because it is never a hit.
- Thresholds are between 0 and 1.
- To tune without exporting again, rerun the same range with new settings. Cached chunks are reused, and scoring runs far faster than real time. The JSON `run.settings` block records the settings for each run so results can be compared. This needs the chunks, so leave out `--cleanup-chunks` until the settings are right.
- Context classes such as `Vehicle` often score above `0.8` while `Engine` on the same window is much lower. A scan does not treat the global top class as the hit. The peak on an event is the highest smoothed score among target classes that passed their own threshold.

There is no flag to turn a context class into a hit.

## What a scan does

1. Validate every flag, check that `ffmpeg` is installed, load and hash-check the model, and, with `--confirm`, check `ANTHROPIC_API_KEY`. Nothing is exported until these pass.
2. Start `npx -y @mgcrea/mcp-unifi-protect@1.1.0` over stdio (or `PROTECT_AUDIO_SEARCH_MCP_COMMAND`).
3. Resolve `--camera` and warn if its current settings say it does not record audio; older recordings may still contain sound.
4. Export the range in chunks (default 10 minutes, last chunk shorter). Transient server errors (429, 5xx) retry with backoff. Client timeouts are not retried immediately because the server may still be writing. A chunk over the download cap is split. Another error skips that chunk.
5. As each chunk lands, extract 16 kHz mono audio and score each window with the TFLite YAMNet model. A chunk with no audio stream is skipped, with a warning. The last partial window of a chunk is scored too. `WindowScorer` is a small interface so another backend could be added later. No ONNX backend ships.
6. Merge target-class frames into events. Smoothing does not average across a gap in the audio.
7. Optionally write clips and snapshots, optionally send snapshots to Anthropic, write the results, then optionally delete the footage chunks.

JSON (`--json`) follows [docs/scan-result.schema.json](docs/scan-result.schema.json) (`schema_version` 2). The file has `start_utc`, `end_utc`, `tool_version`, and a `run` block with the settings, chunk counts, and the chunks that had no audio, failed, or had less audio than their range. Each event's `start` and `end` are in `--tz` with a numeric offset. `confirmation` is `null` when `--confirm` is off. CSV (`--csv`) always includes the confirmation columns, empty when confirmation is off. A cell that starts with `=`, `+`, `-`, or `@` gets a leading `'` so a spreadsheet does not run it as a formula. The terminal table is plain aligned text.

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
- **Exports are slow and large.** Hours of footage take time and disk. A five hour range at the default chunk size is thirty downloads, up to 200 MB each at full quality.
- **Recording gaps.** A chunk with less audio than its range (motion-only recording, say) is flagged in `run.chunks_short_audio`. Times inside it are approximate, because the server reports the start as the end minus the file's length.
- **Private API.** The MCP server wraps Protect's private API, which can change between Protect releases.
- **Times are whole seconds.** `--from` and `--to` drop fractions of a second. A clock time that daylight saving skips or repeats is refused. Pass an offset instead.
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
- Delete footage when you no longer need it. `--cleanup-chunks` removes exported chunks and leaves the review folder. It does not decide your retention period. Footage exported by other tools through the same MCP server may still sit in its `UNIFI_PROTECT_SNAPSHOT_DIR` (default `~/.cache/unifi-protect`). This tool does not write there.
- Prefer a view-only Protect account used only for this tool.
- Do not commit `.env`, footage, or model files.

## Security and contributing

[SECURITY.md](SECURITY.md) covers reporting, secrets, and what the process is allowed to call. [CONTRIBUTING.md](CONTRIBUTING.md) covers the local checks.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
