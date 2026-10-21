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
        parsed = parsed.replace(tzinfo=tz)
    return parsed.astimezone(ZoneInfo("UTC"))


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
