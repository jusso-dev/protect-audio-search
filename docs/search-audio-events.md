# Search as an MCP tool

## Decision

Do not proceed. This note, and the prototype behind `python -m protect_audio_search.search_mcp`, are the end of the idea for this epic. The entry point is not a supported interface. The README does not advertise it.

Reasons:

1. A real multi-hour Protect scan has not been run. The CLI hits are not proven on the footage this tool is for. Freezing an assistant-facing schema before that run would lock in the wrong shape.
2. A five-hour range is thirty exports. `run_scan` exports every chunk before it reports progress, and each export can take a long time. Holding one MCP call open across that work will be cut off by the client. The prototype returns a job id instead, but the job lives only in memory.
3. A supported tool needs a durable job directory, progress on each export chunk, cancellation, and a cap on stored jobs. That is a second product. The CLI already scans.
4. The useful assistant step is reviewing frames. Those results must be file paths, never image bytes. This prototype does not write clips or snapshots. Adding them here would copy the CLI.

The prototype is here so the decision sits next to a concrete schema and a job loop.

## What the prototype is

On-demand. The process starts when you launch the entry point. There is no schedule and no always-on listener. A tool call runs the same scan as the CLI: default target classes, default event settings, chunk length 10 minutes. Credentials stay in the process environment. The tools do not take passwords and do not print them. A failed job keeps the redacted error text.

One scan runs at a time. A second `search_audio_events` call fails until the running job finishes or fails. Finished jobs stay in memory until the process exits. Restarting drops them.

## Tools

The fields below are the contract. The Python MCP SDK also adds `title` values on the generated schema. Those titles are not part of the contract.

### search_audio_events

Start a scan. The call returns as soon as the job is recorded. It does not wait for footage export.

| Field | Required | Default | Meaning |
|---|---|---|---|
| `camera` | yes | | Camera name or id, same rule as the CLI |
| `start` | yes | | `YYYY-MM-DD HH:MM`, or an ISO 8601 value. A naive value is read in `tz` |
| `end` | yes | | Exclusive end, same format. Must be after `start` |
| `tz` | no | `Australia/Sydney` | Zone for naive times and for event times in the result |
| `output_dir` | no | `exports` | Footage chunk directory |

There is no threshold, clip, snapshot, or confirm argument. Those stay on the CLI.

The response uses the status shape:

```json
{
  "job_id": "abc123abc123",
  "state": "running",
  "phase": "export",
  "chunks_total": 30,
  "chunk_index": 0,
  "event_count": null,
  "message": ""
}
```

`chunks_total` comes from the chunk plan and is known before export starts. `chunk_index` stays 0 during export. Today's `run_scan` progress callback runs only after every chunk has been exported, when a chunk is about to be scored. `phase` then becomes `score`, and `chunk_index` is that chunk, counting from 1. A long export therefore looks idle. That gap is one reason not to proceed.

### search_audio_events_status

Input: `job_id` (string, required).

| Field | Meaning |
|---|---|
| `state` | `running`, `succeeded`, or `failed` |
| `phase` | `export`, `score`, `done`, or `failed` |
| `chunks_total` | Planned chunks |
| `chunk_index` | `0` during export, then the chunk being scored, then `chunks_total` when done |
| `event_count` | `null` until the job succeeds |
| `message` | Empty, or the redacted failure text |

### search_audio_events_result

Input: `job_id` (string, required).

A job that is still running, or a job that failed, returns an error. The error text is the redacted message. On success:

```json
{
  "job_id": "abc123abc123",
  "scan": {
    "schema_version": 1
  }
}
```

`scan` is the same object the CLI writes for `--json`. The schema is [scan-result.schema.json](scan-result.schema.json). `clip` and `snapshots` are empty because this prototype does not write review files. The tool does not embed JPEG bytes. To review frames, use the CLI flags `--clips` and `--snapshots` and open the files on disk.

## Long-running work

Do not hold the scan inside one tool call. A client timeout is shorter than a multi-hour export. An MCP progress notification on that same call has the same limit: it dies when the call dies.

The prototype polls. The client calls `search_audio_events`, then `search_audio_events_status`, then `search_audio_events_result`. There is no cancel tool. Stopping the process is the only way to stop a scan.

## How to run it

Use the same machine setup as a CLI scan: Python 3.11, ffmpeg, Node, and the MCP server environment variables. Do not put secrets in tool arguments.

```bash
python -m protect_audio_search.search_mcp
```

The server speaks stdio MCP. It is not an HTTP listener. Point a client at that command only if the limits above are acceptable.

## If this decision is revisited

Proceed only after a real Protect scan produces reviewable hits, and only with all of the following:

- durable jobs on disk, with a bound on how many are kept
- progress for each export chunk, not only for scoring
- cancellation that stops the export in progress
- clip and snapshot paths in the result, still with no image bytes in the tool response
- the same read-only Protect tools as the CLI, and no always-on mode

Until then, use `protect-audio-search scan`.
