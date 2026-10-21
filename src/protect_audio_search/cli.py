"""Command line for listing cameras, listing classes, and scanning footage."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from protect_audio_search.audio import AudioError, prepare_scan
from protect_audio_search.classifier import (
    DEFAULT_CONTEXT_CLASSES,
    DEFAULT_TARGET_CLASSES,
    ModelHashError,
    UnknownClassError,
)
from protect_audio_search.events import add_event_arguments, detect_events, event_config_from_args
from protect_audio_search.footage import (
    DEFAULT_CHUNK_MINUTES,
    ProtectClientError,
    list_cameras_sync,
    redact,
)
from protect_audio_search.output import ScanResult, render_table, write_csv, write_json
from protect_audio_search.scan import format_plan, plan_chunks, run_scan
from protect_audio_search.times import DEFAULT_TIMEZONE, format_zoned, parse_user_time, zone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protect-audio-search",
        description="Search UniFi Protect footage audio for revving, burnouts and tyre squeal.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("cameras", help="List cameras from the Protect console")
    subcommands.add_parser(
        "classes",
        help="List target class names and context class names",
    )

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
    add_event_arguments(scan)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run `cameras`, `classes`, or `scan`. Exit 0 with or without hits."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cameras":
            _cameras()
        elif args.command == "classes":
            _classes()
        else:
            _scan(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except (AudioError, ProtectClientError, ModelHashError, UnknownClassError) as exc:
        print(redact(str(exc)), file=sys.stderr)
        raise SystemExit(1) from exc


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


def _scan(args: argparse.Namespace) -> None:
    tz = zone(args.tz)
    start = parse_user_time(args.from_time, tz)
    end = parse_user_time(args.to_time, tz)
    if args.chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be a positive number of minutes")
    if end <= start:
        raise ValueError("end must be after start")
    config = event_config_from_args(args)
    # Reject bad detection settings before any footage export.
    detect_events([], camera=args.camera, config=config)
    if args.dry_run:
        print(format_plan(plan_chunks(start, end, args.chunk_minutes), tz))
        return

    prepare_scan()

    def progress(index: int, count: int, chunk_start: datetime, chunk_end: datetime) -> None:
        print(
            f"chunk {index}/{count} "
            f"{format_zoned(chunk_start, tz)} to {format_zoned(chunk_end, tz)}",
            file=sys.stderr,
        )

    events = run_scan(
        camera=args.camera,
        start=start,
        end=end,
        output_dir=args.output_dir,
        event_config=config,
        chunk_minutes=args.chunk_minutes,
        progress=progress,
    )
    result = ScanResult(
        camera=args.camera,
        timezone=args.tz,
        start=start,
        end=end,
        events=tuple(events),
    )
    print(render_table(result, tz))
    if args.json_path is not None:
        write_json(args.json_path, result, tz)
    if args.csv_path is not None:
        write_csv(args.csv_path, result, tz)


if __name__ == "__main__":
    main()
