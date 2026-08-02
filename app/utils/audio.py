"""Audio helpers shared by providers and transports."""

from __future__ import annotations

import io
import wave
from contextlib import contextmanager
from pathlib import Path
import tempfile

import soundfile as sf


def waveform_to_wav_bytes(waveform, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return buffer.getvalue()


def wav_duration_seconds(wav_bytes: bytes, sample_rate: int | None = None) -> float:
    """Best-effort duration from WAV bytes; falls back to size estimate."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    if sample_rate and sample_rate > 0:
        # Rough PCM16 mono estimate excluding header
        return max(0.0, (len(wav_bytes) - 44) / (2.0 * sample_rate))
    return 0.0


@contextmanager
def temp_audio_file(audio_bytes: bytes, suffix: str = ".wav"):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        yield tmp.name
    finally:
        Path(tmp.name).unlink(missing_ok=True)
