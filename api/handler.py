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

    if isinstance(num_step, bool) or not isinstance(num_step, (int, float)):
        raise ValueError("num_step must be an integer between 1 and 128")
    num_step = int(num_step)
    if not (1 <= num_step <= 128):
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


def normalize_job_input(raw: Any) -> dict[str, Any]:
    """Accept RunPod / console payload variants and return a flat params dict.

    Supported shapes (after RunPod unwraps the outer job envelope):
      {"input": "hello"}                          # text only
      {"input": "hello", "speaker": "Mohamed"}    # OpenAI-style
      {"text": "hello", "speaker": "Mohamed"}     # legacy
      {"input": {"input": "hello", ...}}          # double-wrapped curl body in console
    """
    if raw is None:
        raise ValueError(
            "input is required. Send {\"input\": {\"input\": \"your text\", \"speaker\": \"Mohamed\"}}"
        )
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("input text is empty")
        return {"input": text}
    if not isinstance(raw, dict):
        raise ValueError(
            f"input must be a JSON object or string, got {type(raw).__name__}"
        )

    payload = dict(raw)
    nested = payload.get("input")
    if isinstance(nested, dict):
        # Console pasted the full curl body → job.input.input is another object.
        payload = {**payload, **nested}
    return payload


def _extract_text(job_input: dict[str, Any]) -> str:
    text = job_input.get("input")
    if isinstance(text, dict):
        text = text.get("input") or text.get("text")
    if text is None:
        text = job_input.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "input is required (string). Example: "
            '{"input": {"input": "ازيك عامل ايه؟", "speaker": "Mohamed", "language": "arz"}}'
        )
    return text.strip()


def job_input_to_speech_request(job_input: dict[str, Any]) -> SpeechRequest:
    """Map OpenAI-compatible (+ extensions / legacy) job input to SpeechRequest."""
    settings = get_settings()
    text = _extract_text(job_input)

    params = _parse_generation_params(job_input)
    model = _optional_str(job_input.get("model"), "model")
    voice = _optional_str(job_input.get("voice"), "voice")
    speaker = _optional_str(job_input.get("speaker"), "speaker")
    explicit_mode = job_input.get("mode")
    instruct = _optional_str(job_input.get("instruct"), "instruct")
    ref_audio_b64 = job_input.get("ref_audio")
    ref_text = _optional_str(job_input.get("ref_text"), "ref_text")

    mode: SpeechMode
    ref_audio: bytes | None = None
    engine = get_speech_engine()

    def _is_builtin_speaker(name: str | None) -> bool:
        if not name:
            return False
        return engine.has_speaker(name)

    if ref_audio_b64 is not None:
        mode = "clone"
        ref_audio = _decode_ref_audio(ref_audio_b64)
    elif explicit_mode in {"auto", "design", "clone"}:
        mode = explicit_mode  # type: ignore[assignment]
        if mode == "design":
            if not instruct and not speaker:
                # voice may be a design instruct string or a built-in speaker name
                if voice and _is_builtin_speaker(voice):
                    speaker = voice
                    mode = "auto"
                else:
                    instruct = instruct or voice
                    if not instruct:
                        raise ValueError("instruct (or voice/speaker) is required for design mode")
            elif speaker and not instruct:
                mode = "auto"
        if mode == "clone":
            raise ValueError("ref_audio is required for clone mode")
    elif instruct:
        mode = "design"
    elif speaker:
        mode = "auto"
    elif voice and voice.lower() in {"default", "auto"}:
        mode = "auto"
        speaker = settings.DEFAULT_SPEAKER
    elif voice and _is_builtin_speaker(voice):
        mode = "auto"
        speaker = voice
    elif voice:
        mode = "design"
        instruct = voice
    else:
        mode = "auto"
        speaker = settings.DEFAULT_SPEAKER

    return SpeechRequest(
        text=text.strip(),
        mode=mode,
        model=model,
        voice=voice,
        speaker=speaker,
        instruct=instruct,
        ref_audio=ref_audio,
        ref_text=ref_text,
        **params,
    )


def _summarize_input(raw: Any) -> str:
    if isinstance(raw, str):
        preview = raw[:80] + ("…" if len(raw) > 80 else "")
        return f"str len={len(raw)} preview={preview!r}"
    if isinstance(raw, dict):
        return f"dict keys={sorted(raw.keys())}"
    return type(raw).__name__


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    engine = get_speech_engine()
    if not engine.is_ready():
        return {"error": engine.get_load_error() or "Model is not ready"}

    raw_input = job.get("input")
    logger.info("Job %s received payload: %s", job.get("id"), _summarize_input(raw_input))

    try:
        job_input = normalize_job_input(raw_input)
        request = job_input_to_speech_request(job_input)
        logger.info(
            "Synthesizing mode=%s speaker=%s chars=%s",
            request.mode,
            request.speaker,
            len(request.text),
        )
        result = await engine.generate(request)
        logger.info(
            "Synthesis complete: %.2fs audio, %s bytes",
            result.duration_seconds,
            len(result.audio),
        )
        return {
            "audio_base64": base64.b64encode(result.audio).decode("ascii"),
            "sample_rate": result.sample_rate,
            "content_type": result.content_type,
        }
    except QueueFullError as exc:
        logger.warning("Job rejected: %s", exc)
        return {"error": str(exc)}
    except RequestTimeoutError as exc:
        logger.warning("Job timed out: %s", exc)
        return {"error": str(exc)}
    except ValueError as exc:
        logger.warning("Job validation failed: %s", exc)
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
