# protect-audio-search

An on-demand command line tool that pulls recorded footage from a UniFi Protect system through an existing MCP server, then searches the audio for sounds like cars revving, burnouts and tyre squeal. It reports timestamped hits, with optional trimmed clips and snapshots for each one.

> Status: planning. There is no implementation yet. See the issues for the work breakdown.

## Why

UniFi Protect's built-in audio detections cover smoke and CO alarms, sirens, car horns, car alarms, glass breaking, barking, speech and baby crying. There is no category for revving, burnouts or tyre squeal. LLM assistants such as Claude can look at images but cannot listen to audio, so this tool uses a sound classifier to do the listening and leaves visual review of the frames to a person or, optionally, an LLM.

This is not an always-on listener. You run it for a camera and a time range, and it tells you what it heard.

## How it will work

1. **Footage.** The CLI is a Python MCP client (official `mcp` SDK, stdio transport). It launches [mgcrea/mcp-unifi-protect](https://github.com/mgcrea/mcp-unifi-protect) (for example `npx -y @mgcrea/mcp-unifi-protect`) and calls `unifi_protect_list_cameras` and `unifi_protect_export_video`, which writes MP4s to disk. The launch command is configurable so other servers can be swapped in. [sirkirby/unifi-mcp](https://github.com/sirkirby/unifi-mcp) is a fine server for interactive use, but its clip export returns metadata rather than a file on disk, so it is not the default here.
2. **Chunking.** The requested range is split into chunks (default 10 minutes), each exported separately to keep files small and allow resuming.
3. **Audio.** `ffmpeg` extracts 16 kHz mono PCM audio from each chunk. Cameras without an audio track are warned about and skipped.
4. **Classification.** [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) (trained on AudioSet, 521 classes) scores roughly 0.96 s windows with a 0.48 s hop, via TFLite or ONNX runtime.
5. **Events.** Frames whose score for a target class passes a threshold are merged into events, with a minimum duration and a gap tolerance.
6. **Output.** A table in the terminal plus JSON or CSV. Times are shown in `Australia/Sydney` by default (configurable). Optional trimmed clips and snapshots per hit.
7. **Optional visual check.** Snapshots for each hit can be sent to an LLM to confirm a vehicle or tyre smoke and attempt to read a plate. Off by default.

### Default target classes

These are the exact YAMNet / AudioSet display names used by default:

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

Context classes, reported but not treated as hits on their own: `Vehicle` (294), `Motor vehicle (road)` (300), `Car` (301), `Car passing by` (308), `Idling` (346), `Engine starting` (345), `Traffic noise, roadway noise` (321).

## Configuration

Credentials come from environment variables only, never flags or config files checked into git. The MCP server reads its own variables (for mgcrea/mcp-unifi-protect: `UNIFI_PROTECT_HOST`, `UNIFI_PROTECT_USERNAME`, `UNIFI_PROTECT_PASSWORD` and friends, see its README). An optional LLM key is read from the provider's standard variable.

## Honest limitations

- **It only hears what the camera hears.** Many cameras have no microphone or have it disabled. Distance, wind, rain and wall reflections all reduce accuracy.
- **YAMNet is a general model.** It was not trained for hoon detection specifically. Expect false positives (motorbikes, leaf blowers, trucks, loud music) and missed events. Thresholds will need tuning per camera.
- **Exports are slow.** Pulling hours of footage from an NVR takes time and disk space. Long ranges are expensive.
- **Private API.** mgcrea/mcp-unifi-protect wraps Protect's private API, which can change between Protect releases.
- **No identification guarantee.** Plate reading from a night snapshot is unreliable. Treat any LLM output as a hint, not evidence.

## Privacy and legal

Recording audio and number plates of people on public streets is regulated. In Australia, surveillance device and listening device laws vary by state and territory, and recording private conversations can be an offence. Check the rules where you live before enabling audio on a camera or keeping clips. Keep footage local, keep it only as long as you need it, and do not send snapshots to a third-party LLM unless you are comfortable with that provider handling them.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
