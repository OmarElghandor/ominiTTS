"""RunPod Serverless adapter — thin transport over SpeechEngine.

No provider / OmniVoice imports. Business logic lives in app.speech.
"""

from __future__ import annotations

import base64
import binascii
import logging
import sys
from pathlib import Path
from typing import Any

import runpod
import torch

# Ensure /app is on sys.path when executed as /app/handler.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import get_settings
from app.speech import (
    QueueFullError,
    RequestTimeoutError,
    SpeechMode,
    SpeechRequest,
    get_speech_engine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("omnivoice.api.handler")


def _optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    stripped = value.strip()
    return stripped or None


def _decode_ref_audio(ref_audio: Any) -> bytes:
    if not isinstance(ref_audio, str) or not ref_audio.strip():
        raise ValueError("ref_audio must be valid base64-encoded audio data")
    try:
        return base64.b64decode(ref_audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("ref_audio must be valid base64-encoded audio data") from exc


def _parse_generation_params(job_input: dict[str, Any]) -> dict[str, Any]:
    num_step = job_input.get("num_step", 32)
    speed = job_input.get("speed", 1.0)
    duration = job_input.get("duration")
    language = _optional_str(job_input.get("language"), "language")

    if not isinstance(num_step, int) or isinstance(num_step, bool) or not (1 <= num_step <= 128):
        raise ValueError("num_step must be an integer between 1 and 128")
    if not isinstance(speed, (int, float)) or isinstance(speed, bool) or not (0 < float(speed) <= 5.0):
        raise ValueError("speed must be a number in (0, 5]")
    if duration is not None:
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or float(duration) <= 0:
            raise ValueError("duration must be a positive number when provided")
        duration = float(duration)

    return {
        "num_step": num_step,
        "speed": float(speed),
        "duration": duration,
        "language": language,
    }


def job_input_to_speech_request(job_input: dict[str, Any]) -> SpeechRequest:
    """Map OpenAI-compatible (+ extensions / legacy) job input to SpeechRequest."""
    text = job_input.get("input")
    if text is None:
        text = job_input.get("text")  # legacy alias
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input is required (OpenAI-compatible text field)")

    params = _parse_generation_params(job_input)
    model = _optional_str(job_input.get("model"), "model")
    voice = _optional_str(job_input.get("voice"), "voice")
    explicit_mode = job_input.get("mode")
    instruct = _optional_str(job_input.get("instruct"), "instruct")
    ref_audio_b64 = job_input.get("ref_audio")
    ref_text = _optional_str(job_input.get("ref_text"), "ref_text")

    mode: SpeechMode
    ref_audio: bytes | None = None

    if ref_audio_b64 is not None:
        mode = "clone"
        ref_audio = _decode_ref_audio(ref_audio_b64)
    elif explicit_mode in {"auto", "design", "clone"}:
        mode = explicit_mode  # type: ignore[assignment]
        if mode == "design":
            instruct = instruct or voice
            if not instruct:
                raise ValueError("instruct (or voice) is required for design mode")
        if mode == "clone":
            raise ValueError("ref_audio is required for clone mode")
    elif instruct:
        mode = "design"
    elif voice and voice.lower() not in {"default", "auto"}:
        mode = "design"
        instruct = voice
    else:
        mode = "auto"

    return SpeechRequest(
        text=text.strip(),
        mode=mode,
        model=model,
        voice=voice,
        instruct=instruct,
        ref_audio=ref_audio,
        ref_text=ref_text,
        **params,
    )


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    engine = get_speech_engine()
    if not engine.is_ready():
        return {"error": engine.get_load_error() or "Model is not ready"}

    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return {"error": "input must be a JSON object"}

    try:
        request = job_input_to_speech_request(job_input)
        result = await engine.generate(request)
        return {
            "audio_base64": base64.b64encode(result.audio).decode("ascii"),
            "sample_rate": result.sample_rate,
            "content_type": result.content_type,
        }
    except QueueFullError as exc:
        return {"error": str(exc)}
    except RequestTimeoutError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except torch.cuda.OutOfMemoryError as exc:
        logger.error("CUDA OOM: %s", exc)
        engine.clear_cuda_cache()
        return {"error": "Service temporarily overloaded. Try again shortly."}
    except Exception as exc:
        logger.exception("Unhandled handler error")
        return {"error": f"Internal error: {type(exc).__name__}: {exc}"}


def main() -> None:
    settings = get_settings()
    engine = get_speech_engine()
    logger.info("Starting RunPod serverless worker (fail-fast load + warmup)...")
    engine.startup(settings, fail_fast=True)
    if not engine.is_ready():
        raise SystemExit(f"Worker not ready: {engine.status()}")
    logger.info("Worker ready: %s", engine.status())
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
