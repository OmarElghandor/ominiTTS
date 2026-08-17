"""OmniVoice SpeechProvider implementation."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import torch
from omnivoice import OmniVoice

from app.config import Settings
from app.model_store import assert_model_store_verified
from app.speech.types import SpeechRequest, SpeechResult
from app.utils.audio import temp_audio_file, wav_duration_seconds, waveform_to_wav_bytes

logger = logging.getLogger(__name__)

_HF_DOWNLOAD_GUARD_MSG = (
    "HF download attempted in API process — run scripts/bootstrap_model.py to seed the volume"
)


def _block_hf_download(*_args, **_kwargs):
    raise RuntimeError(_HF_DOWNLOAD_GUARD_MSG)


class OmniVoiceProvider:
    name = "omnivoice"

    def __init__(self) -> None:
        self._model: OmniVoice | None = None
        self._device: str = "cpu"
        self._tokenizer_ok: bool = False

    def load(self, settings: Settings) -> None:
        resolved_device, resolved_dtype = settings.resolve_device_and_dtype()
        self._device = resolved_device

        settings.assert_model_store_ready()
        model_path = str(settings.resolve_model_path())
        assert_model_store_verified(settings.resolve_model_path())

        logger.info(
            "Loading OmniVoice from local store %s on device=%s dtype=%s",
            model_path,
            resolved_device,
            resolved_dtype,
        )

        with patch("huggingface_hub.snapshot_download", side_effect=_block_hf_download):
            self._model = OmniVoice.from_pretrained(
                model_path,
                device_map=resolved_device,
                dtype=resolved_dtype,
                load_asr=False,
            )

        # Tokenizer files are verified by model_store; mark loaded after successful from_pretrained.
        self._tokenizer_ok = True
        logger.info(
            "OmniVoice model loaded successfully (sample_rate=%s)",
            self._model.sampling_rate,
        )

    def warmup(self, text: str) -> None:
        if self._model is None:
            raise RuntimeError("OmniVoice model is not loaded")
        result = self.generate(
            SpeechRequest(text=text, mode="auto", num_step=16, speed=1.0)
        )
        logger.info(
            "Warmup complete: %s chars -> %.2fs audio",
            len(text),
            result.duration_seconds,
        )

    def generate(self, request: SpeechRequest) -> SpeechResult:
        if self._model is None:
            raise RuntimeError("OmniVoice model is not loaded")

        kwargs = self._build_generate_kwargs(request)

        if request.mode == "clone":
            if not request.ref_audio:
                raise ValueError("ref_audio is required for clone mode")
            if request.ref_text is None:
                logger.warning(
                    "ref_text omitted for voice cloning — auto-transcription requires ASR "
                    "(load_asr=False offline; provide ref_text for clone requests)"
                )
            with temp_audio_file(request.ref_audio, suffix=request.ref_audio_suffix) as ref_path:
                kwargs["ref_audio"] = ref_path
                if request.ref_text is not None:
                    kwargs["ref_text"] = request.ref_text
                audios = self._model.generate(**kwargs)
        else:
            audios = self._model.generate(**kwargs)

        sample_rate = int(self._model.sampling_rate)
        wav_bytes = waveform_to_wav_bytes(audios[0], sample_rate)
        return SpeechResult(
            audio=wav_bytes,
            sample_rate=sample_rate,
            duration_seconds=wav_duration_seconds(wav_bytes, sample_rate),
            content_type="audio/wav",
        )

    def _build_generate_kwargs(self, request: SpeechRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "text": request.text,
            "num_step": request.num_step,
            "speed": request.speed,
        }
        if request.duration is not None:
            kwargs["duration"] = request.duration
        if request.language is not None:
            kwargs["language"] = request.language
        if request.mode == "design":
            if not request.instruct:
                raise ValueError("instruct is required for design mode")
            kwargs["instruct"] = request.instruct
        return kwargs

    @property
    def sample_rate(self) -> int:
        if self._model is None:
            return 24000
        return int(self._model.sampling_rate)

    def is_loaded(self) -> bool:
        return self._model is not None

    def tokenizer_loaded(self) -> bool:
        return self._tokenizer_ok

    @property
    def device(self) -> str:
        return self._device

    def clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
