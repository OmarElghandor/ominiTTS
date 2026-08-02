"""SpeechProvider protocol — implement this to add a new TTS backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.config import Settings
from app.speech.types import SpeechRequest, SpeechResult


@runtime_checkable
class SpeechProvider(Protocol):
    name: str

    def load(self, settings: Settings) -> None:
        """Load model + tokenizer from MODEL_STORE_DIR. Never download remotely."""
        ...

    def warmup(self, text: str) -> None:
        """Run a short inference to warm CUDA/kernels."""
        ...

    def generate(self, request: SpeechRequest) -> SpeechResult:
        """Synthesize speech; return binary audio."""
        ...

    @property
    def sample_rate(self) -> int:
        ...

    def is_loaded(self) -> bool:
        ...

    def tokenizer_loaded(self) -> bool:
        ...

    def clear_cuda_cache(self) -> None:
        ...
