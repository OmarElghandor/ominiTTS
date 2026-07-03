import io
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from omnivoice import OmniVoice

from app.config import Settings

logger = logging.getLogger(__name__)

_model: OmniVoice | None = None
_model_ready: bool = False
_model_loading: bool = False
_load_error: str | None = None
_resolved_device: str = "cpu"


def is_model_ready() -> bool:
    return _model_ready


def is_model_loading() -> bool:
    return _model_loading


def get_load_error() -> str | None:
    return _load_error


def get_resolved_device() -> str:
    return _resolved_device


def _set_load_error(message: str) -> None:
    global _load_error, _model_loading
    _load_error = message
    _model_loading = False


def initialize_device(settings: Settings) -> None:
    global _resolved_device
    resolved_device, _ = settings.resolve_device_and_dtype()
    _resolved_device = resolved_device


def get_model() -> OmniVoice | None:
    return _model


def load_model(settings: Settings) -> None:
    global _model, _model_ready, _model_loading, _load_error, _resolved_device

    _model_loading = True
    _load_error = None

    resolved_device, resolved_dtype = settings.resolve_device_and_dtype()
    _resolved_device = resolved_device

    cache_dir = Path(settings.HF_HOME)
    logger.info(
        "Loading OmniVoice model '%s' on device=%s dtype=%s (cache=%s)",
        settings.MODEL_NAME,
        resolved_device,
        resolved_dtype,
        cache_dir,
    )

    if not settings.hf_cache_has_content():
        logger.warning(
            "HF cache at %s looks empty — first load will download multi-GB weights from "
            "Hugging Face. This can take 10–30+ minutes on CPU. Check /readyz before calling TTS.",
            cache_dir,
        )

    try:
        _model = OmniVoice.from_pretrained(
            settings.MODEL_NAME,
            device_map=resolved_device,
            dtype=resolved_dtype,
            load_asr=True,
        )
        _model_ready = True
        _model_loading = False
        logger.info("OmniVoice model loaded successfully (sample_rate=%s)", _model.sampling_rate)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        _set_load_error(message)
        logger.exception("OmniVoice model load failed")
        raise


def _build_generate_kwargs(
    *,
    text: str,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
    language: str | None = None,
    num_step: int = 32,
    speed: float = 1.0,
    duration: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "text": text,
        "num_step": num_step,
        "speed": speed,
    }
    if duration is not None:
        kwargs["duration"] = duration
    if language is not None:
        kwargs["language"] = language
    if ref_audio_path is not None:
        kwargs["ref_audio"] = ref_audio_path
    if ref_text is not None:
        kwargs["ref_text"] = ref_text
    if instruct is not None:
        kwargs["instruct"] = instruct
    return kwargs


def _waveform_to_wav_bytes(waveform, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return buffer.getvalue()


@contextmanager
def _temp_audio_file(audio_bytes: bytes, suffix: str = ".wav"):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        yield tmp.name
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def synthesize_clone(
    *,
    text: str,
    ref_audio_bytes: bytes,
    ref_text: str | None = None,
    language: str | None = None,
    num_step: int = 32,
    speed: float = 1.0,
    duration: float | None = None,
    ref_audio_suffix: str = ".wav",
) -> bytes:
    if _model is None or not _model_ready:
        raise RuntimeError("Model is not loaded")

    if ref_text is None:
        logger.warning(
            "ref_text omitted for voice cloning — OmniVoice will auto-transcribe via Whisper (slower)"
        )

    with _temp_audio_file(ref_audio_bytes, suffix=ref_audio_suffix) as ref_path:
        kwargs = _build_generate_kwargs(
            text=text,
            ref_audio_path=ref_path,
            ref_text=ref_text,
            language=language,
            num_step=num_step,
            speed=speed,
            duration=duration,
        )
        audios = _model.generate(**kwargs)

    return _waveform_to_wav_bytes(audios[0], _model.sampling_rate)


def synthesize_design(
    *,
    text: str,
    instruct: str,
    language: str | None = None,
    num_step: int = 32,
    speed: float = 1.0,
    duration: float | None = None,
) -> bytes:
    if _model is None or not _model_ready:
        raise RuntimeError("Model is not loaded")

    kwargs = _build_generate_kwargs(
        text=text,
        instruct=instruct,
        language=language,
        num_step=num_step,
        speed=speed,
        duration=duration,
    )
    audios = _model.generate(**kwargs)
    return _waveform_to_wav_bytes(audios[0], _model.sampling_rate)


def synthesize_auto(
    *,
    text: str,
    language: str | None = None,
    num_step: int = 32,
    speed: float = 1.0,
    duration: float | None = None,
) -> bytes:
    if _model is None or not _model_ready:
        raise RuntimeError("Model is not loaded")

    kwargs = _build_generate_kwargs(
        text=text,
        language=language,
        num_step=num_step,
        speed=speed,
        duration=duration,
    )
    audios = _model.generate(**kwargs)
    return _waveform_to_wav_bytes(audios[0], _model.sampling_rate)


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
