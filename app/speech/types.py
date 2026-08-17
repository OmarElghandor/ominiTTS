"""Shared speech request/response types (provider-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SpeechMode = Literal["auto", "design", "clone"]


@dataclass(slots=True)
class SpeechRequest:
    text: str
    mode: SpeechMode = "auto"
    language: str | None = None
    speed: float = 1.0
    num_step: int = 32
    duration: float | None = None
    instruct: str | None = None
    speaker: str | None = None
    ref_audio: bytes | None = None
    ref_text: str | None = None
    ref_audio_suffix: str = ".wav"
    voice: str | None = None
    model: str | None = None
    request_id: str | None = None
    normalize: bool = True


@dataclass(slots=True)
class SpeechResult:
    """Binary-oriented synthesis result for any transport (HTTP WAV, base64, stream)."""

    audio: bytes
    sample_rate: int
    duration_seconds: float
    content_type: str = "audio/wav"
    extras: dict = field(default_factory=dict)


class QueueFullError(Exception):
    """Raised when MAX_QUEUE_SIZE waiters are already pending."""


class RequestTimeoutError(Exception):
    """Raised when REQUEST_TIMEOUT is exceeded."""
