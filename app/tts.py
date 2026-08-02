"""Deprecated compatibility shim — prefer app.speech.SpeechEngine."""

from __future__ import annotations

import torch

from app.speech.engine import get_speech_engine
from app.speech.model_manager import get_model_manager


def is_model_ready() -> bool:
    return get_speech_engine().is_ready()


def is_model_loading() -> bool:
    return get_speech_engine().is_loading()


def get_load_error() -> str | None:
    return get_speech_engine().get_load_error()


def get_resolved_device() -> str:
    return get_speech_engine().get_device()


def clear_cuda_cache() -> None:
    get_speech_engine().clear_cuda_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_manager_status():
    return get_model_manager().status()
