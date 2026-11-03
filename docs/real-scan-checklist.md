# Real five-hour scan checklist

The automated tests use a mock MCP server and short fixtures. The epic's last acceptance check is a scan over a real five-hour Protect range that produces hits a person can review. This is the run to do on a machine that can reach the console.

## Before

1. Install as in the README, then `protect-audio-search --version`.
2. Set the `UNIFI_PROTECT_*` variables for a local view-only account in the shell. Leave `UNIFI_PROTECT_ALLOW_WRITES` unset.
3. `protect-audio-search cameras` and pick a camera that faces a road.
4. `protect-audio-search check --camera "NAME"`. Every line should start with `ok` or `note`. Fix anything that fails before you export.
5. Pick a five-hour window where you know, or suspect, revving or a burnout happened. Write down the times you expect.
6. `protect-audio-search scan --camera "NAME" --from ... --to ... --dry-run` and check the thirty chunk times are the clock you meant.

## A short run first

Scan 30 minutes that contain one known sound, with `--clips --snapshots 3`. Open the clip and listen.

- No event: lower that class with `--class-threshold`, or lower `--min-duration`.
- Too many events: raise `--threshold`.
- `Chunk ... failed` mentioning a byte limit: the chunk was split. If splits are frequent, pass `--channel 1`.

Rerunning the same range reuses the exported chunks, so tuning is quick.

## The five-hour run

```bash
protect-audio-search scan \
  --camera "NAME" \
  --from "YYYY-MM-DD 21:00" \
  --to "YYYY-MM-DD 02:00" \
  --clips --snapshots 3 \
  --json five-hours.json --csv five-hours.csv
```

Leave it running. `export N/30` and `chunk N/30` lines show where it is. Ctrl-C keeps the chunks, and the same command resumes.

## What counts as done

- The final stderr line says `Scored 30 of 30 exported chunks`, or the JSON `run.chunks_failed` explains each gap.
- `run.chunks_without_audio` is empty.
- Each event's clip plays, and its time matches the sound in the clip within a second or two.
- At least one event is a real revving, burnout, or squeal that a person can confirm from the clip.
- False positives are noted with the settings used (`run.settings`), so the defaults can be tuned.

Record the camera type, Protect version, settings, event count, and true or false positive count in the epic. Do not attach footage, plates, or snapshots to a public issue.
