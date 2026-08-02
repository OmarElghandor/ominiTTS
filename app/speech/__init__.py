"""Speech synthesis facade — SpeechEngine + ModelManager + providers."""

from app.speech.engine import SpeechEngine, get_speech_engine
from app.speech.model_manager import ModelManager, get_model_manager
from app.speech.types import (
    QueueFullError,
    RequestTimeoutError,
    SpeechMode,
    SpeechRequest,
    SpeechResult,
)

__all__ = [
    "ModelManager",
    "QueueFullError",
    "RequestTimeoutError",
    "SpeechEngine",
    "SpeechMode",
    "SpeechRequest",
    "SpeechResult",
    "get_model_manager",
    "get_speech_engine",
]
