"""Optional vision check. Off unless the CLI is given `--confirm`.

Snapshots are sent to Anthropic only when confirmation is on. Those images
leave this machine. The API key is read from the environment and is never
printed.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_MAX_IMAGES = 3
DEFAULT_MAX_CALLS = 20
DEFAULT_MAX_COST_USD = 0.50
# Short-prompt Haiku 5.5 rates, USD per token. Not a bill.
_INPUT_USD_PER_TOKEN = 0.10 / 1_000_000
_OUTPUT_USD_PER_TOKEN = 0.50 / 1_000_000

_PROMPT = (
    "Look at these camera frames. Reply with one JSON object and no other text. "
    "Keys: vehicle_present (boolean), smoke_present (boolean), "
    "plate_text (string or null), plate_confidence (number from 0 to 1, or null), "
    "note (a short string). This is a hint for a person, not evidence."
)


class ConfirmError(Exception):
    """The vision provider failed. The message must not include the API key."""


@dataclass(frozen=True)
class Confirmation:
    vehicle_present: bool | None
    smoke_present: bool | None
    plate_text: str | None
    plate_confidence: float | None
    note: str


@dataclass(frozen=True)
class ConfirmBudget:
    max_images_per_event: int = DEFAULT_MAX_IMAGES
    max_calls: int = DEFAULT_MAX_CALLS
    max_cost_usd: float = DEFAULT_MAX_COST_USD


@dataclass(frozen=True)
class ConfirmSummary:
    calls: int
    images: int
    estimated_usd: float
    skipped: int

    def render(self) -> str:
        return (
            f"Vision calls: {self.calls}. "
            f"Images sent: {self.images}. "
            f"Estimated cost: USD {self.estimated_usd:.4f}. "
            f"Skipped: {self.skipped}."
        )


@dataclass(frozen=True)
class ProviderReply:
    text: str
    estimated_usd: float


class VisionProvider(Protocol):
    def complete(self, images: Sequence[bytes]) -> ProviderReply:
        """Send images and return the model text plus an estimated cost."""


@dataclass
class _Tally:
    calls: int = 0
    images: int = 0
    estimated_usd: float = 0.0
    skipped: int = 0


def confirm_events(
    snapshot_groups: Sequence[Sequence[Path]],
    provider: VisionProvider,
    budget: ConfirmBudget,
) -> tuple[list[Confirmation], ConfirmSummary]:
    """Confirm each event. Stops calling once the call or cost cap is hit."""
    if budget.max_images_per_event < 1:
        raise ValueError("confirm image cap must be at least 1")
    if budget.max_calls < 1:
        raise ValueError("confirm call cap must be at least 1")
    if budget.max_cost_usd < 0:
        raise ValueError("confirm cost cap must be zero or positive")
    tally = _Tally()
    results: list[Confirmation] = []
    for paths in snapshot_groups:
        chosen = [path for path in paths if path.is_file()][: budget.max_images_per_event]
        if not chosen:
            tally.skipped += 1
            results.append(_note("Skipped. This event has no snapshots."))
            continue
        if tally.calls >= budget.max_calls or tally.estimated_usd >= budget.max_cost_usd:
            tally.skipped += 1
            results.append(_note("Skipped. The confirmation cost cap was reached."))
            continue
        reply = provider.complete([path.read_bytes() for path in chosen])
        tally.calls += 1
        tally.images += len(chosen)
        tally.estimated_usd += max(0.0, reply.estimated_usd)
        results.append(parse_confirmation(reply.text))
    return results, ConfirmSummary(
        calls=tally.calls,
        images=tally.images,
        estimated_usd=tally.estimated_usd,
        skipped=tally.skipped,
    )


def parse_confirmation(text: str) -> Confirmation:
    """Read a model reply. A bad reply becomes an empty hint, not an exception."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return _unreadable()
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return _unreadable()
    if not isinstance(payload, dict):
        return _unreadable()
    note = payload.get("note")
    plate = payload.get("plate_text")
    return Confirmation(
        vehicle_present=_optional_bool(payload.get("vehicle_present")),
        smoke_present=_optional_bool(payload.get("smoke_present")),
        plate_text=plate if isinstance(plate, str) and plate else None,
        plate_confidence=_optional_score(payload.get("plate_confidence")),
        note=note if isinstance(note, str) else "",
    )


def format_notice() -> str:
    return "Vision confirmation sends snapshot images to Anthropic. They leave this machine."


def build_anthropic_provider() -> AnthropicVision:
    """Build the Anthropic provider. The key stays in the client."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Vision confirmation needs it, and the value is never printed."
        )
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return AnthropicVision(api_key=key, model=model)


class AnthropicVision:
    """Claude Haiku vision. Constructed only when `--confirm` is on."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._api_key = api_key
        self._model = model

    def complete(self, images: Sequence[bytes]) -> ProviderReply:
        import anthropic

        blocks: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(image).decode("ascii"),
                },
            }
            for image in images
        ]
        blocks.append({"type": "text", "text": _PROMPT})
        payload: Any = [{"role": "user", "content": blocks}]
        try:
            message = anthropic.Anthropic(api_key=self._api_key).messages.create(
                model=self._model,
                max_tokens=400,
                messages=payload,
            )
        except Exception as exc:
            raise ConfirmError(f"Anthropic request failed ({type(exc).__name__}).") from None
        parts = [block.text for block in message.content if block.type == "text"]
        usage = message.usage
        return ProviderReply(
            text="\n".join(parts),
            estimated_usd=estimate_cost_usd(usage.input_tokens, usage.output_tokens),
        )


def estimate_cost_usd(input_tokens: float, output_tokens: float) -> float:
    """Short-prompt Haiku 5.5 estimate. Not an invoice."""
    return input_tokens * _INPUT_USD_PER_TOKEN + output_tokens * _OUTPUT_USD_PER_TOKEN


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    if 0.0 <= score <= 1.0:
        return score
    return None


def _unreadable() -> Confirmation:
    return _note("The vision response could not be read.")


def _note(note: str) -> Confirmation:
    return Confirmation(
        vehicle_present=None,
        smoke_present=None,
        plate_text=None,
        plate_confidence=None,
        note=note,
    )
