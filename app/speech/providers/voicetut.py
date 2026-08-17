"""VoiceTut-TTS SpeechProvider — Egyptian Arabic fine-tune of OmniVoice."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import torch

from app.config import Settings
from app.model_store import assert_model_store_verified
from app.speech.types import SpeechRequest, SpeechResult
from app.utils.audio import temp_audio_file, wav_duration_seconds, waveform_to_wav_bytes

logger = logging.getLogger(__name__)

_HF_DOWNLOAD_GUARD_MSG = (
    "HF download attempted in API process — run scripts/bootstrap_model.py to seed the volume"
)

_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}


def _block_hf_download(*_args, **_kwargs):
    raise RuntimeError(_HF_DOWNLOAD_GUARD_MSG)


class VoiceTutProvider:
    name = "voicetut"

    def __init__(self) -> None:
        self._tts: Any = None
        self._device: str = "cpu"
        self._tokenizer_ok: bool = False
        self._default_speaker: str = "Mohamed"
        self._default_language: str = "arz"

    def load(self, settings: Settings) -> None:
        from voicetut_tts import VoiceTutTTS

        resolved_device, resolved_dtype = settings.resolve_device_and_dtype()
        self._device = resolved_device
        self._default_speaker = settings.DEFAULT_SPEAKER
        self._default_language = "arz"

        settings.assert_model_store_ready()
        model_path = str(settings.resolve_model_path())
        assert_model_store_verified(settings.resolve_model_path())

        dtype_str = _DTYPE_TO_STR.get(resolved_dtype, "float16")
        logger.info(
            "Loading VoiceTut-TTS from local store %s on device=%s dtype=%s",
            model_path,
            resolved_device,
            dtype_str,
        )

        with patch("huggingface_hub.snapshot_download", side_effect=_block_hf_download):
            self._tts = VoiceTutTTS.from_pretrained(
                model_path,
                device=resolved_device,
                dtype=dtype_str,
                language=self._default_language,
                load_asr=False,
            )

        self._tokenizer_ok = True
        speakers = self.list_speakers()
        logger.info(
            "VoiceTut-TTS loaded (sample_rate=%s, speakers=%d)",
            self._tts.sampling_rate,
            len(speakers),
        )

    def warmup(self, text: str) -> None:
        if self._tts is None:
            raise RuntimeError("VoiceTut-TTS model is not loaded")
        result = self.generate(
            SpeechRequest(
                text=text,
                mode="auto",
                speaker=self._default_speaker,
                language=self._default_language,
                num_step=16,
                speed=1.0,
            )
        )
        logger.info(
            "Warmup complete: %s chars -> %.2fs audio (speaker=%s)",
            len(text),
            result.duration_seconds,
            self._default_speaker,
        )

    def generate(self, request: SpeechRequest) -> SpeechResult:
        if self._tts is None:
            raise RuntimeError("VoiceTut-TTS model is not loaded")

        language = request.language or self._default_language
        speaker, instruct = self._resolve_speaker_and_instruct(request)
        param_overrides = self._build_param_overrides(request)

        if request.mode == "clone":
            if not request.ref_audio:
                raise ValueError("ref_audio is required for clone mode")
            if request.ref_text is None:
                logger.warning(
                    "ref_text omitted for voice cloning — auto-transcription requires ASR "
                    "(load_asr=False offline; provide ref_text for clone requests)"
                )
            with temp_audio_file(request.ref_audio, suffix=request.ref_audio_suffix) as ref_path:
                wav = self._tts.synthesize(
                    request.text,
                    ref_audio=ref_path,
                    ref_text=request.ref_text,
                    language=language,
                    normalize=request.normalize,
                    **param_overrides,
                )
        else:
            wav = self._tts.synthesize(
                request.text,
                speaker=speaker,
                instruct=instruct,
                language=language,
                normalize=request.normalize,
                **param_overrides,
            )

        sample_rate = int(self._tts.sampling_rate)
        wav_bytes = waveform_to_wav_bytes(wav, sample_rate)
        return SpeechResult(
            audio=wav_bytes,
            sample_rate=sample_rate,
            duration_seconds=wav_duration_seconds(wav_bytes, sample_rate),
            content_type="audio/wav",
        )

    def _resolve_speaker_and_instruct(
        self, request: SpeechRequest
    ) -> tuple[str | None, str | None]:
        """Pick exactly one of speaker / instruct for non-clone modes."""
        if request.mode == "clone":
            return None, None

        speaker = (request.speaker or "").strip() or None
        instruct = (request.instruct or "").strip() or None

        if request.mode == "design":
            if instruct:
                return None, instruct
            if speaker:
                return speaker, None
            raise ValueError("instruct (or speaker) is required for design mode")

        # auto: prefer explicit speaker, else default built-in voice
        if instruct and speaker:
            raise ValueError("Choose ONE of: speaker or instruct")
        if instruct:
            return None, instruct
        return speaker or self._default_speaker, None

    @staticmethod
    def _build_param_overrides(request: SpeechRequest) -> dict[str, Any]:
        overrides: dict[str, Any] = {
            "num_step": request.num_step,
            "speed": request.speed,
        }
        if request.duration is not None:
            overrides["duration"] = request.duration
        return overrides

    def list_speakers(self) -> list[dict[str, Any]]:
        if self._tts is None:
            return []
        speakers = self._tts.list_speakers()
        return [
            {
                "speaker_id": spk.speaker_id,
                "speaker_name": spk.speaker_name,
                "gender": spk.gender,
                "tags": list(spk.tags),
            }
            for spk in speakers
        ]

    def has_speaker(self, name_or_id: str) -> bool:
        if self._tts is None or not self._tts.registry:
            return False
        try:
            self._tts.registry.get(name_or_id)
            return True
        except KeyError:
            return False

    @property
    def sample_rate(self) -> int:
        if self._tts is None:
            return 24000
        return int(self._tts.sampling_rate)

    def is_loaded(self) -> bool:
        return self._tts is not None

    def tokenizer_loaded(self) -> bool:
        return self._tokenizer_ok

    @property
    def device(self) -> str:
        return self._device

    def clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
