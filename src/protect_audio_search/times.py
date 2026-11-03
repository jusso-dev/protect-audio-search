"""Parse and render scan times. Storage is UTC. Display uses the chosen zone."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TIMEZONE = "Australia/Sydney"


def zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone {name!r}.") from exc


def parse_user_time(text: str, tz: ZoneInfo) -> datetime:
    """Parse a user time. Naive values are read in `tz`. Result is UTC."""
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse time {text!r}. Use 'YYYY-MM-DD HH:MM' or an ISO 8601 value."
        ) from exc
    if parsed.tzinfo is None:
        parsed = _localise(parsed, tz, text)
    # Exports and chunk names work in whole seconds.
    return parsed.astimezone(ZoneInfo("UTC")).replace(microsecond=0)


def _localise(naive: datetime, tz: ZoneInfo, text: str) -> datetime:
    """Attach `tz`, refusing a clock time that is skipped or repeated by a DST change."""
    first = naive.replace(tzinfo=tz, fold=0)
    second = naive.replace(tzinfo=tz, fold=1)
    round_trip = first.astimezone(ZoneInfo("UTC")).astimezone(tz).replace(tzinfo=None)
    if round_trip != naive:
        raise ValueError(
            f"{text!r} does not exist in {tz.key}: the clocks skip it for daylight saving. "
            "Pass a time with an offset, for example 2026-10-04T03:30+11:00."
        )
    if first.utcoffset() != second.utcoffset():
        raise ValueError(
            f"{text!r} happens twice in {tz.key} when daylight saving ends. "
            "Pass a time with an offset, for example 2026-04-05T02:30+11:00."
        )
    return first


def format_zoned(value: datetime, tz: ZoneInfo) -> str:
    """Render an aware datetime in `tz` with the numeric offset."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(tz).isoformat(timespec="seconds")


def format_utc(value: datetime) -> str:
    utc = value.astimezone(ZoneInfo("UTC"))
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def add_seconds(value: datetime, seconds: float) -> datetime:
    return value + timedelta(seconds=seconds)
