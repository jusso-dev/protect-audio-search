"""Command line entry point.

`scan` accepts the event-detection parameters. Footage download and
classification are wired in later issues.
"""

from __future__ import annotations

import argparse
import sys

from protect_audio_search.events import add_event_arguments, event_config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protect-audio-search",
        description=("Search UniFi Protect footage audio for revving, burnouts and tyre squeal."),
    )
    subcommands = parser.add_subparsers(dest="command")
    scan = subcommands.add_parser(
        "scan",
        help="Search a camera recording for revving, burnouts and tyre squeal",
    )
    add_event_arguments(scan)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments. `scan` checks detection flags and reports they are set."""
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        config = event_config_from_args(args)
        print(
            "Detection settings:"
            f" threshold={config.threshold}"
            f" gap={config.gap_seconds}"
            f" min_duration={config.min_duration_seconds}"
            f" smooth_frames={config.smooth_frames}",
            file=sys.stdout,
        )


if __name__ == "__main__":
    main()
