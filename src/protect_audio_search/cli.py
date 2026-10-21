"""Command line entry point.

Feature subcommands land in later issues. This module only makes `--help` work.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="protect-audio-search",
        description=("Search UniFi Protect footage audio for revving, burnouts and tyre squeal."),
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments. `--help` exits after printing usage."""
    build_parser().parse_args(argv)


if __name__ == "__main__":
    main()
