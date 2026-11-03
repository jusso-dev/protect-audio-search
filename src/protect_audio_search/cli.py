"""Command line for listing cameras, listing classes, checking setup, and scanning footage."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from protect_audio_search import __version__
from protect_audio_search.audio import AudioError, prepare_scan
from protect_audio_search.classifier import (
    DEFAULT_CONTEXT_CLASSES,
    DEFAULT_TARGET_CLASSES,
    ModelCacheError,
    ModelHashError,
    UnknownClassError,
    default_cache_dir,
    load_classifier,
    suggest_class_names,
)
from protect_audio_search.clips import (
    DEFAULT_PADDING_SECONDS,
    EventMedia,
    capture_events,
    delete_chunks,
    review_directory,
)
from protect_audio_search.confirm import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_COST_USD,
    DEFAULT_MAX_IMAGES,
    AnthropicVision,
    Confirmation,
    ConfirmBudget,
    ConfirmError,
    build_anthropic_provider,
    confirm_events,
    format_notice,
)
from protect_audio_search.events import (
    EventConfig,
    add_event_arguments,
    detect_events,
    event_config_from_args,
)
from protect_audio_search.footage import (
    DEFAULT_CHANNEL,
    DEFAULT_CHUNK_MINUTES,
    ProtectClientError,
    list_cameras_sync,
    redact,
    resolve_camera,
)
from protect_audio_search.output import (
    ChunkNote,
    RunSummary,
    ScanResult,
    render_table,
    write_csv,
    write_json,
)
from protect_audio_search.scan import ScanOutput, format_plan, plan_chunks, run_scan
from protect_audio_search.times import DEFAULT_TIMEZONE, format_zoned, parse_user_time, zone

MAX_CHUNK_MINUTES = 24 * 60
MAX_PADDING_SECONDS = 600.0
MAX_SNAPSHOTS = 20
MAX_EVENT_SECONDS = 3600.0
_FAILURES = (
    AudioError,
    ProtectClientError,
    ModelHashError,
    ModelCacheError,
    UnknownClassError,
    ConfirmError,
    OSError,
    subprocess.SubprocessError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protect-audio-search",
        description="Search UniFi Protect footage audio for revving, burnouts and tyre squeal.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("cameras", help="List cameras from the Protect console")
    subcommands.add_parser(
        "classes",
        help="List target class names and context class names",
    )
    check = subcommands.add_parser(
        "check",
        help="Check ffmpeg, the model, the console, and a camera's audio before a scan",
        description=(
            "Check what a scan needs without exporting footage: ffmpeg, the YAMNet model, "
            "the MCP server and console login, and that the camera records audio."
        ),
    )
    check.add_argument("--camera", help="Camera name or id to check")

    scan = subcommands.add_parser(
        "scan",
        help="Search a camera recording for revving, burnouts and tyre squeal",
        description=(
            "Export footage, score the audio, and list events. "
            "A time without a timezone is read in --tz and stored in UTC. "
            "Printed times use --tz."
        ),
    )
    scan.add_argument("--camera", required=True, help="Camera name or id")
    scan.add_argument(
        "--from",
        dest="from_time",
        required=True,
        metavar="TIME",
        help="Range start. Example: 2026-10-20 21:00",
    )
    scan.add_argument(
        "--to",
        dest="to_time",
        required=True,
        metavar="TIME",
        help="Range end, exclusive. Same format as --from",
    )
    scan.add_argument(
        "--tz",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone for naive times and displayed results (default {DEFAULT_TIMEZONE})",
    )
    scan.add_argument(
        "--chunk-minutes",
        type=int,
        default=DEFAULT_CHUNK_MINUTES,
        help=f"Footage chunk length in minutes (default {DEFAULT_CHUNK_MINUTES})",
    )
    scan.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        choices=range(4),
        metavar="{0,1,2,3}",
        help=(
            f"Protect encoder channel to export (default {DEFAULT_CHANNEL}, full quality). "
            "A higher channel is a smaller file and a softer picture"
        ),
    )
    scan.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory for exported footage chunks (default exports)",
    )
    scan.add_argument(
        "--json",
        type=Path,
        dest="json_path",
        metavar="PATH",
        help="JSON file. See docs/scan-result.schema.json",
    )
    scan.add_argument(
        "--csv",
        type=Path,
        dest="csv_path",
        metavar="PATH",
        help="Write one CSV row per event",
    )
    scan.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned chunks. Do not export footage or classify audio",
    )
    scan.add_argument(
        "--clips",
        action="store_true",
        help="Trim an MP4 for each event from the local chunks",
    )
    scan.add_argument(
        "--clip-padding",
        type=float,
        default=DEFAULT_PADDING_SECONDS,
        help=f"Seconds of padding before and after each clip (default {DEFAULT_PADDING_SECONDS:g})",
    )
    scan.add_argument(
        "--snapshots",
        type=int,
        default=0,
        metavar="N",
        help=f"Save N evenly spaced JPEG frames for each event, up to {MAX_SNAPSHOTS}. 0 is off",
    )
    scan.add_argument(
        "--cleanup-chunks",
        action="store_true",
        help="Delete the exported footage chunks when the run finishes",
    )
    scan.add_argument(
        "--confirm",
        action="store_true",
        help="Images leave this machine. See docs/vision-confirmation.md",
    )
    scan.add_argument(
        "--confirm-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help=f"Most images to send per event (default {DEFAULT_MAX_IMAGES})",
    )
    scan.add_argument(
        "--confirm-max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"Most vision calls in one run (default {DEFAULT_MAX_CALLS})",
    )
    scan.add_argument(
        "--confirm-max-cost",
        type=float,
        default=DEFAULT_MAX_COST_USD,
        help=f"Stop once estimated cost reaches this many USD (default {DEFAULT_MAX_COST_USD})",
    )
    add_event_arguments(scan)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a subcommand. A scan exits 0 with or without hits."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cameras":
            _cameras()
        elif args.command == "classes":
            _classes()
        elif args.command == "check":
            _check(args)
        else:
            _scan(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except _FAILURES as exc:
        print(redact(str(exc)), file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print(
            "Interrupted. Exported chunks are kept. Run the same command again to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None


def _cameras() -> None:
    cameras = list_cameras_sync()
    if not cameras:
        print("No cameras.")
        return
    for camera in cameras:
        print(f"{camera.id}\t{camera.name}")


def _classes() -> None:
    print("Targets:")
    for name in DEFAULT_TARGET_CLASSES:
        print(f"  {name}")
    print("Context:")
    for name in DEFAULT_CONTEXT_CLASSES:
        print(f"  {name}")


def _check(args: argparse.Namespace) -> None:
    """Preflight. Each line is `ok`, or a note. A failure exits 1 with the reason."""
    prepare_scan()
    print("ok    ffmpeg and ffprobe are on PATH")
    load_classifier(class_names=list(DEFAULT_TARGET_CLASSES))
    print(f"ok    YAMNet model and class map in {default_cache_dir()} match their hashes")
    cameras = list_cameras_sync()
    print(f"ok    MCP server started and listed {len(cameras)} cameras")
    if args.camera is None:
        return
    camera = resolve_camera(cameras, args.camera)
    if camera.is_connected is False:
        print(f"note  {camera.name!r} is offline now. Recorded footage can still export.")
    if camera.recording_mode is not None:
        print(f"note  {camera.name!r} recording mode is {camera.recording_mode}")
    if camera.records_audio is False:
        print(
            f"note  {camera.name!r} does not currently record audio; "
            "historical footage may still contain sound"
        )
    elif camera.records_audio is None:
        print(f"note  the server did not say whether {camera.name!r} records audio")
    else:
        print(f"ok    {camera.name!r} ({camera.id}) records audio")


def _scan(args: argparse.Namespace) -> None:
    tz = zone(args.tz)
    start = parse_user_time(args.from_time, tz)
    end = parse_user_time(args.to_time, tz)
    _validate_scan_args(args)
    if end <= start:
        raise ValueError("end must be after start")
    config = event_config_from_args(args)
    _check_class_thresholds(config)
    # Reject bad detection settings before any footage export.
    detect_events([], camera=args.camera, config=config)
    if args.dry_run:
        print(format_plan(plan_chunks(start, end, args.chunk_minutes), tz))
        return

    # Footage, clips, and snapshots are private. Keep new files to this user.
    os.umask(0o077)
    prepare_scan()
    provider: AnthropicVision | None = build_anthropic_provider() if args.confirm else None
    for path in (args.json_path, args.csv_path):
        if path is not None:
            _check_writable(path)

    def progress(index: int, count: int, chunk_start: datetime, chunk_end: datetime) -> None:
        print(
            f"chunk {index}/{count} "
            f"{format_zoned(chunk_start, tz)} to {format_zoned(chunk_end, tz)}",
            file=sys.stderr,
        )

    def export_progress(
        index: int, count: int, chunk_start: datetime, chunk_end: datetime, cached: bool
    ) -> None:
        action = "cached" if cached else "export"
        print(
            f"{action} {index}/{count} "
            f"{format_zoned(chunk_start, tz)} to {format_zoned(chunk_end, tz)}",
            file=sys.stderr,
        )

    scanned = run_scan(
        camera=args.camera,
        start=start,
        end=end,
        output_dir=args.output_dir,
        event_config=config,
        chunk_minutes=args.chunk_minutes,
        channel=args.channel,
        progress=progress,
        export_progress=export_progress,
    )
    wants_media = bool(args.clips or args.snapshots)
    run_dir = review_directory(args.output_dir, args.camera, start).resolve()
    if wants_media:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=run_dir))
    media: list[EventMedia] = []
    if wants_media:
        media = capture_events(
            scanned.events,
            scanned.chunks,
            run_dir=run_dir,
            padding_seconds=args.clip_padding,
            write_clips=args.clips,
            snapshot_count=args.snapshots,
        )
        for number, item in enumerate(media, start=1):
            if item.error:
                print(f"Event {number:04d}: {item.error}", file=sys.stderr)
    confirmations: tuple[Confirmation | None, ...] = ()
    if provider is not None:
        print(format_notice(), file=sys.stderr)
        confirmed, summary = confirm_events(
            [item.snapshots for item in media],
            provider,
            ConfirmBudget(
                max_images_per_event=args.confirm_images,
                max_calls=args.confirm_max_calls,
                max_cost_usd=args.confirm_max_cost,
            ),
        )
        confirmations = tuple(confirmed)
        print(summary.render(), file=sys.stderr)
    result = ScanResult(
        camera=args.camera,
        timezone=args.tz,
        start=start,
        end=end,
        events=tuple(scanned.events),
        clip_paths=tuple(None if item.clip is None else item.clip.as_posix() for item in media),
        snapshot_paths=tuple(tuple(path.as_posix() for path in item.snapshots) for item in media),
        confirmations=confirmations,
        media_errors=tuple(item.error for item in media),
        run=_summary(args, config, scanned, run_dir if wants_media else None),
    )
    print(render_table(result, tz))
    # Results are written before cleanup, so a write error never costs the footage.
    if args.json_path is not None:
        write_json(args.json_path, result, tz)
    if args.csv_path is not None:
        write_csv(args.csv_path, result, tz)
    if wants_media:
        write_json(run_dir / "results.json", result, tz)
        print(f"Review files: {run_dir}", file=sys.stderr)
    print(_coverage(scanned), file=sys.stderr)
    incomplete = bool(
        not scanned.scored
        or scanned.failed
        or scanned.silent
        or scanned.short
        or any(item.error for item in media)
    )
    if args.cleanup_chunks and not incomplete:
        removed = delete_chunks(scanned.chunks)
        print(f"Deleted {removed} footage chunks.", file=sys.stderr)
    elif args.cleanup_chunks:
        print(
            "Kept footage chunks because the scan or media processing was incomplete.",
            file=sys.stderr,
        )
    if scanned.scored == 0:
        _no_audio(bool(scanned.failed))
    if scanned.failed:
        print("Scan incomplete: one or more chunks failed or were not attempted.", file=sys.stderr)
        raise SystemExit(3)


def _validate_scan_args(args: argparse.Namespace) -> None:
    if not 1 <= args.chunk_minutes <= MAX_CHUNK_MINUTES:
        raise ValueError(f"chunk minutes must be between 1 and {MAX_CHUNK_MINUTES}")
    if not math.isfinite(args.clip_padding) or not 0 <= args.clip_padding <= MAX_PADDING_SECONDS:
        raise ValueError(f"clip padding must be between 0 and {MAX_PADDING_SECONDS:g} seconds")
    if not 0 <= args.snapshots <= MAX_SNAPSHOTS:
        raise ValueError(f"snapshot count must be between 0 and {MAX_SNAPSHOTS}")
    if args.confirm and args.snapshots < 1:
        raise ValueError("Pass --snapshots N with --confirm. The images are those snapshots.")
    if args.confirm_images < 1:
        raise ValueError("confirm image cap must be at least 1")
    if args.confirm_max_calls < 1:
        raise ValueError("confirm call cap must be at least 1")
    if not math.isfinite(args.confirm_max_cost) or args.confirm_max_cost < 0:
        raise ValueError("confirm cost cap must be zero or positive")
    for label, value in (("gap", args.gap_seconds), ("min duration", args.min_duration_seconds)):
        if not math.isfinite(value) or not 0 <= value <= MAX_EVENT_SECONDS:
            raise ValueError(f"{label} must be between 0 and {MAX_EVENT_SECONDS:g} seconds")


def _check_class_thresholds(config: EventConfig) -> None:
    """Only target classes are scored. Any other name would be silently ignored."""
    for name in config.class_thresholds:
        if name in DEFAULT_TARGET_CLASSES:
            continue
        if name in DEFAULT_CONTEXT_CLASSES:
            raise ValueError(
                f"{name!r} is a context class. It is never a hit, so a threshold has no effect."
            )
        close = suggest_class_names(name, list(DEFAULT_TARGET_CLASSES))
        raise ValueError(
            f"Unknown target class {name!r} in --class-threshold. "
            f"Close matches: {', '.join(close or DEFAULT_TARGET_CLASSES)}."
        )


def _check_writable(path: Path) -> None:
    if path.is_dir():
        raise ValueError(f"{path} is a directory. Pass a file path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(path.parent, os.W_OK):
        raise ValueError(f"Cannot write to {path.parent}.")


def _summary(
    args: argparse.Namespace,
    config: EventConfig,
    scanned: ScanOutput,
    review_dir: Path | None,
) -> RunSummary:
    return RunSummary(
        settings={
            "threshold": config.threshold,
            "class_thresholds": dict(config.class_thresholds),
            "smooth_frames": config.smooth_frames,
            "gap_seconds": config.gap_seconds,
            "min_duration_seconds": config.min_duration_seconds,
            "chunk_minutes": args.chunk_minutes,
            "channel": args.channel,
        },
        chunks_planned=scanned.planned,
        chunks_scored=scanned.scored,
        without_audio=tuple(ChunkNote(chunk.start, chunk.end) for chunk in scanned.silent),
        failed=tuple(ChunkNote(item.start, item.end, item.error) for item in scanned.failed),
        short_audio=tuple(ChunkNote(chunk.start, chunk.end) for chunk in scanned.short),
        review_dir=None if review_dir is None else review_dir.as_posix(),
    )


def _coverage(scanned: ScanOutput) -> str:
    text = (
        f"Scored {scanned.scored} of {len(scanned.chunks)} exported chunks. "
        f"No audio: {len(scanned.silent)}. Failed: {len(scanned.failed)}. "
        f"Events: {len(scanned.events)}."
    )
    if scanned.failed:
        text += " Run the same command again to retry the failed chunks."
    return text


def _no_audio(failed: bool = False) -> NoReturn:
    if failed:
        print(
            "No audio was scored. Check the reported chunk failures and retry the scan.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(
        "No audio was scored, so an empty result does not mean nothing happened. "
        "Check that the camera has a microphone, that it is on, and that Protect keeps "
        "audio in recordings: protect-audio-search check --camera NAME",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
