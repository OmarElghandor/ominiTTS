"""Speech provider registry."""

from __future__ import annotations

from app.speech.provider import SpeechProvider
from app.speech.providers.omnivoice import OmniVoiceProvider
from app.speech.providers.voicetut import VoiceTutProvider


def create_provider(name: str) -> SpeechProvider:
    key = (name or "voicetut").strip().lower()
    if key in {"voicetut", "voicetut-tts", "voicetut_tts"}:
        return VoiceTutProvider()
    if key == "omnivoice":
        return OmniVoiceProvider()
    raise ValueError(
        f"Unknown SPEECH_PROVIDER={name!r}. Supported: voicetut, omnivoice"
    )
