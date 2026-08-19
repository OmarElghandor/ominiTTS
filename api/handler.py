"""RunPod Serverless adapter — thin transport over SpeechEngine.

No provider / OmniVoice imports. Business logic lives in app.speech.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
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
    settings = get_settings()
    num_step = job_input.get("num_step", settings.DEFAULT_NUM_STEP)
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


_TTS_PARAM_KEYS = frozenset(
    {
        "text",
        "input",
        "speaker",
        "voice",
        "language",
        "instruct",
        "mode",
        "model",
        "ref_audio",
        "ref_text",
        "num_step",
        "speed",
        "duration",
    }
)


def _spoken_text_from_dict(payload: dict[str, Any]) -> str | None:
    """Return the first non-empty spoken-text string from common field names."""
    for key in ("text", "input"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            inner = val.get("text") or val.get("input")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return None


def normalize_job_input(raw: Any) -> dict[str, Any]:
    """Accept RunPod / console payload variants and return a flat params dict.

    Supported shapes (after RunPod unwraps the outer job envelope):
      {"text": "hello", "speaker": "Mohamed"}     # Requests tab
      {"input": "hello"}                          # text only
      {"input": "hello", "speaker": "Mohamed"}    # OpenAI-style
      {"input": {"input": "hello", ...}}          # double-wrapped curl body
      {"input": "", "text": "hello", ...}         # empty input + text alias
    """
    if raw is None:
        raise ValueError(
            'text or input is required. Example: {"text": "ازيك؟", "speaker": "Mohamed", "language": "arz"}'
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
        payload = {**payload, **nested}

    spoken = _spoken_text_from_dict(payload)
    if spoken is None:
        raise ValueError(
            'text or input is required (non-empty string). Example: '
            '{"text": "صباح الخير", "speaker": "Mohamed", "language": "arz"}'
        )
    payload["input"] = spoken
    return payload


def _extract_text(job_input: dict[str, Any]) -> str:
    spoken = _spoken_text_from_dict(job_input)
    if spoken is None:
        raise ValueError(
            'text or input is required (non-empty string). Example: '
            '{"text": "صباح الخير", "speaker": "Mohamed", "language": "arz"}'
        )
    return spoken


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


_JOB_ENVELOPE_KEYS = {"id", "input", "webhook"}


def _text_preview(text: str, limit: int = 80) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "…"


def _summarize_input(raw: Any) -> str:
    if isinstance(raw, str):
        return f"str len={len(raw)} preview={_text_preview(raw)!r}"
    if isinstance(raw, dict):
        keys = sorted(raw.keys())
        spoken = _spoken_text_from_dict(raw)
        if spoken:
            return f"dict keys={keys} text={_text_preview(spoken)!r}"
        return f"dict keys={keys}"
    return type(raw).__name__


def _raw_job_payload(job: dict[str, Any]) -> Any:
    """Prefer job['input']; merge top-level TTS fields when input is empty."""
    top = {key: value for key, value in job.items() if key not in _JOB_ENVELOPE_KEYS}
    raw = job.get("input")

    if raw is None:
        return top or None

    if isinstance(raw, dict):
        merged = {**raw, **top}
        if _spoken_text_from_dict(merged) is None and top:
            logger.info(
                "Job %s merging top-level fields into input: %s",
                job.get("id"),
                sorted(top),
            )
        return merged

    if isinstance(raw, str):
        if raw.strip():
            return {**top, "input": raw.strip()}
        return top or None

    return raw


def _ensure_job_input(data: Any) -> Any:
    """Give job-take dicts an `input` key so the RunPod SDK accepts them."""
    if isinstance(data, list):
        return [_ensure_job_input(item) for item in data]
    if not isinstance(data, dict):
        return data

    if "id" not in data:
        return data

    top = {
        key: value
        for key, value in data.items()
        if key not in _JOB_ENVELOPE_KEYS
    }
    inp = data.get("input")

    if "input" not in data:
        logger.info(
            "Coerced job %s without input key: keys=%s",
            data.get("id"),
            sorted(top),
        )
        return {"id": data["id"], "input": top}

    if isinstance(inp, dict):
        merged = {**inp, **top}
        if merged != inp:
            logger.info(
                "Coerced job %s merged sibling fields into input: keys=%s",
                data.get("id"),
                sorted(merged),
            )
            return {"id": data["id"], "input": merged}
        return data

    if isinstance(inp, str) and inp.strip():
        return data

    if top:
        logger.info(
            "Coerced job %s empty input replaced with top-level fields: keys=%s",
            data.get("id"),
            sorted(top),
        )
        return {"id": data["id"], "input": top}

    return data


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    engine = get_speech_engine()
    if not engine.is_ready():
        return {"error": engine.get_load_error() or "Model is not ready"}

    raw_input = _raw_job_payload(job)
    logger.info("Job %s received payload: %s", job.get("id"), _summarize_input(raw_input))

    try:
        job_input = normalize_job_input(raw_input)
        request = job_input_to_speech_request(job_input)
        logger.info(
            "Job %s synthesizing mode=%s speaker=%s chars=%s text=%r",
            job.get("id"),
            request.mode,
            request.speaker,
            len(request.text),
            _text_preview(request.text),
        )
        result = await engine.generate(request)
        logger.info(
            "Job %s synthesis complete: %.2fs audio, %s bytes",
            job.get("id"),
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


def _patch_json_on_response(response: Any) -> Any:
    orig_json = response.json

    async def patched_json(*json_args, **json_kwargs):
        data = await orig_json(*json_args, **json_kwargs)
        return _ensure_job_input(data)

    response.json = patched_json
    return response


class _JobTakeGetWrapper:
    """Preserve aiohttp `async with session.get(...)` while coercing job JSON."""

    def __init__(self, cm: Any) -> None:
        self._cm = cm

    def __await__(self):
        return self._await_response().__await__()

    async def _await_response(self):
        response = await self._cm
        return _patch_json_on_response(response)

    async def __aenter__(self):
        response = await self._cm.__aenter__()
        return _patch_json_on_response(response)

    async def __aexit__(self, *exc: Any):
        return await self._cm.__aexit__(*exc)


def _patch_runpod_empty_job_poll() -> None:
    """Accept job-take payloads that have id but no input; ignore empty polls.

    runpod-python 1.7.x requires both `id` and `input`. The Requests tab may
    deliver `{id, text, speaker, ...}` without wrapping TTS fields in `input`.
    Empty FlashBoot polls (`{}`) still lack `id` and are treated as no job.
    """
    try:
        from runpod.serverless.modules import rp_job
    except ImportError:
        return

    original_get_job = rp_job.get_job

    async def get_job_tolerant(session, num_jobs: int = 1):
        orig_get = session.get

        def patched_get(*args, **kwargs):
            return _JobTakeGetWrapper(orig_get(*args, **kwargs))

        try:
            session.get = patched_get
            return await original_get_job(session, num_jobs)
        except Exception as exc:
            if "missing field(s): id or input" in str(exc):
                logger.debug("Ignoring empty job-take payload (FlashBoot/idle poll)")
                return None
            raise
        finally:
            session.get = orig_get

    rp_job.get_job = get_job_tolerant
    try:
        from runpod.serverless.modules import rp_scale

        rp_scale.get_job = get_job_tolerant
    except Exception:
        pass


def main() -> None:
    settings = get_settings()
    engine = get_speech_engine()
    logger.info("Starting RunPod serverless worker (fail-fast load + warmup)...")
    engine.startup(settings, fail_fast=True)
    if not engine.is_ready():
        raise SystemExit(f"Worker not ready: {engine.status()}")
    logger.info("Worker ready: %s", engine.status())
    if os.environ.get("RUNPOD_POD_ID"):
        logger.warning(
            "RunPod serverless: set endpoint Execution timeout to at least 120s "
            "(default 30s causes 'job timed out after 1 retries'). "
            "Endpoint → Edit → Job timeout / Execution timeout."
        )
    _patch_runpod_empty_job_poll()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
