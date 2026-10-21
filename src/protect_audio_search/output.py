"""Terminal table, JSON, and CSV for detection events."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from protect_audio_search.confirm import Confirmation
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
    clip_paths: tuple[str | None, ...] = ()
    snapshot_paths: tuple[tuple[str, ...], ...] = ()
    confirmations: tuple[Confirmation | None, ...] = ()


def _aligned_media(
    result: ScanResult,
) -> tuple[tuple[str | None, ...], tuple[tuple[str, ...], ...]]:
    count = len(result.events)
    clips = result.clip_paths or tuple(None for _ in range(count))
    empty: tuple[str, ...] = ()
    snapshots = result.snapshot_paths or tuple(empty for _ in range(count))
    if len(clips) != count or len(snapshots) != count:
        raise ValueError("clip and snapshot paths must match the event count")
    return clips, snapshots


def _aligned_confirmations(result: ScanResult) -> tuple[Confirmation | None, ...]:
    if not result.confirmations:
        return tuple(None for _ in result.events)
    if len(result.confirmations) != len(result.events):
        raise ValueError("confirmations must match the event count")
    return result.confirmations


def _confirmation_payload(item: Confirmation | None) -> dict[str, object] | None:
    if item is None:
        return None
    return {
        "vehicle_present": item.vehicle_present,
        "smoke_present": item.smoke_present,
        "plate_text": item.plate_text,
        "plate_confidence": item.plate_confidence,
        "note": item.note,
    }


def result_payload(result: ScanResult, tz: ZoneInfo) -> dict[str, object]:
    clips, snapshots = _aligned_media(result)
    confirmations = _aligned_confirmations(result)
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
                "clip": clip,
                "snapshots": list(snaps),
                "confirmation": _confirmation_payload(confirmation),
            }
            for event, clip, snaps, confirmation in zip(
                result.events, clips, snapshots, confirmations, strict=True
            )
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
                "clip",
                "snapshots",
                "vehicle_present",
                "smoke_present",
                "plate_text",
                "plate_confidence",
                "confirm_note",
            ],
        )
        writer.writeheader()
        clips, snapshots = _aligned_media(result)
        confirmations = _aligned_confirmations(result)
        for event, clip, snaps, confirmation in zip(
            result.events, clips, snapshots, confirmations, strict=True
        ):
            writer.writerow(
                {
                    "camera": event.camera,
                    "start": format_zoned(event.start, tz),
                    "end": format_zoned(event.end, tz),
                    "peak_class": event.peak_class,
                    "peak_score": f"{event.peak_score:.3f}",
                    "mean_score": f"{event.mean_score:.3f}",
                    "contributing_classes": "|".join(event.contributing_classes),
                    "clip": clip or "",
                    "snapshots": "|".join(snaps),
                    **_confirm_csv(confirmation),
                }
            )


def _confirm_cells(item: Confirmation | None) -> tuple[str, str, str]:
    if item is None:
        return "", "", ""
    return _yes_no(item.vehicle_present), _yes_no(item.smoke_present), item.plate_text or ""


def _confirm_csv(item: Confirmation | None) -> dict[str, str]:
    if item is None:
        return {
            "vehicle_present": "",
            "smoke_present": "",
            "plate_text": "",
            "plate_confidence": "",
            "confirm_note": "",
        }
    score = "" if item.plate_confidence is None else f"{item.plate_confidence:.3f}"
    return {
        "vehicle_present": _yes_no(item.vehicle_present),
        "smoke_present": _yes_no(item.smoke_present),
        "plate_text": item.plate_text or "",
        "plate_confidence": score,
        "confirm_note": item.note,
    }


def _yes_no(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def render_table(result: ScanResult, tz: ZoneInfo) -> str:
    if not result.events:
        return "No events."
    confirmations = _aligned_confirmations(result)
    show_confirm = any(item is not None for item in confirmations)
    headers = ["start", "end", "class", "score", "mean", "classes"]
    if show_confirm:
        headers.extend(["vehicle", "smoke", "plate"])
    rows: list[list[str]] = []
    for event, confirmation in zip(result.events, confirmations, strict=True):
        row = [
            format_zoned(event.start, tz),
            format_zoned(event.end, tz),
            event.peak_class,
            f"{event.peak_score:.3f}",
            f"{event.mean_score:.3f}",
            ", ".join(event.contributing_classes),
        ]
        if show_confirm:
            vehicle, smoke, plate = _confirm_cells(confirmation)
            row.extend([vehicle, smoke, plate])
        rows.append(row)
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
