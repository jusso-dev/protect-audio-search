"""Terminal table, JSON, and CSV for detection events."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from protect_audio_search.events import DetectionEvent
from protect_audio_search.times import format_utc, format_zoned

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScanResult:
    camera: str
    timezone: str
    start: datetime
    end: datetime
    events: tuple[DetectionEvent, ...]


def result_payload(result: ScanResult, tz: ZoneInfo) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "camera": result.camera,
        "timezone": result.timezone,
        "start_utc": format_utc(result.start),
        "end_utc": format_utc(result.end),
        "events": [
            {
                "camera": event.camera,
                "start": format_zoned(event.start, tz),
                "end": format_zoned(event.end, tz),
                "peak_class": event.peak_class,
                "peak_score": event.peak_score,
                "mean_score": event.mean_score,
                "contributing_classes": list(event.contributing_classes),
            }
            for event in result.events
        ],
    }


def write_json(path: Path, result: ScanResult, tz: ZoneInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_payload(result, tz), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, result: ScanResult, tz: ZoneInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "camera",
                "start",
                "end",
                "peak_class",
                "peak_score",
                "mean_score",
                "contributing_classes",
            ],
        )
        writer.writeheader()
        for event in result.events:
            writer.writerow(
                {
                    "camera": event.camera,
                    "start": format_zoned(event.start, tz),
                    "end": format_zoned(event.end, tz),
                    "peak_class": event.peak_class,
                    "peak_score": f"{event.peak_score:.3f}",
                    "mean_score": f"{event.mean_score:.3f}",
                    "contributing_classes": "|".join(event.contributing_classes),
                }
            )


def render_table(result: ScanResult, tz: ZoneInfo) -> str:
    if not result.events:
        return "No events."
    headers = ("start", "end", "class", "score", "mean", "classes")
    rows = [
        (
            format_zoned(event.start, tz),
            format_zoned(event.end, tz),
            event.peak_class,
            f"{event.peak_score:.3f}",
            f"{event.mean_score:.3f}",
            ", ".join(event.contributing_classes),
        )
        for event in result.events
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    lines.extend(
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers))) for row in rows
    )
    return "\n".join(lines)
