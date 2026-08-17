"""SpeechEngine — provider-agnostic facade with concurrency + metrics."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.metrics.collect import gpu_memory_mb, reset_peak_memory_stats
from app.metrics.logging import get_metrics_logger, log_metrics
from app.speech.model_manager import ModelManager, get_model_manager
from app.speech.types import (
    QueueFullError,
    RequestTimeoutError,
    SpeechRequest,
    SpeechResult,
)

logger = logging.getLogger(__name__)
metrics_logger = get_metrics_logger()


class SpeechEngine:
    def __init__(self, manager: ModelManager | None = None) -> None:
        self._manager = manager or get_model_manager()
        self._settings: Settings | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._waiters = 0
        self._waiters_lock = asyncio.Lock()
        self._cold_start_pending = True
        self._started = False

    def _ensure_semaphore(self, settings: Settings) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_REQUESTS))
        return self._semaphore

    def is_ready(self) -> bool:
        return self._manager.is_ready()

    def is_loading(self) -> bool:
        return self._manager.is_loading()

    def get_load_error(self) -> str | None:
        return self._manager.get_load_error()

    def get_device(self) -> str:
        return self._manager.get_device()

    def status(self) -> dict[str, Any]:
        return self._manager.status()

    def clear_cuda_cache(self) -> None:
        self._manager.clear_cuda_cache()

    def list_speakers(self) -> list[dict]:
        """Return built-in speakers when the active provider supports them."""
        if not self._manager.is_ready():
            return []
        provider = self._manager.provider
        list_fn = getattr(provider, "list_speakers", None)
        if not callable(list_fn):
            return []
        return list(list_fn())

    def has_speaker(self, name_or_id: str) -> bool:
        if not self._manager.is_ready():
            return False
        provider = self._manager.provider
        has_fn = getattr(provider, "has_speaker", None)
        if callable(has_fn):
            return bool(has_fn(name_or_id))
        return False

    def startup(self, settings: Settings | None = None, *, fail_fast: bool = True) -> None:
        """Load model + warmup once. fail_fast=True raises (serverless); False records error."""
        settings = settings or get_settings()
        self._settings = settings
        logging.getLogger().setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        try:
            self._manager.load(settings)
            self._write_ready_marker(settings)
            self._started = True
        except Exception:
            self._remove_ready_marker(settings)
            if fail_fast:
                raise
            # Soft mode: load_error already set on manager
            self._started = False

    def _write_ready_marker(self, settings: Settings) -> None:
        path = Path(settings.READY_MARKER_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ready\n", encoding="utf-8")

    def _remove_ready_marker(self, settings: Settings) -> None:
        Path(settings.READY_MARKER_PATH).unlink(missing_ok=True)

    async def generate(self, request: SpeechRequest) -> SpeechResult:
        settings = self._settings or get_settings()
        if not self._manager.is_ready():
            err = self._manager.get_load_error() or "Model is not ready"
            raise RuntimeError(err)

        text = request.text.strip()
        if not text:
            raise ValueError("text must not be empty")
        if len(text) > settings.MAX_TEXT_LENGTH:
            raise ValueError(
                f"text exceeds maximum length of {settings.MAX_TEXT_LENGTH} characters"
            )
        request.text = text
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        async with self._waiters_lock:
            if self._waiters >= settings.MAX_QUEUE_SIZE:
                raise QueueFullError(
                    f"Queue full: {self._waiters} waiters (MAX_QUEUE_SIZE={settings.MAX_QUEUE_SIZE})"
                )
            self._waiters += 1

        semaphore = self._ensure_semaphore(settings)
        total_t0 = time.perf_counter()
        queue_wait_ms = 0.0
        inference_ms = 0.0
        cold_start = False

        try:
            queue_t0 = time.perf_counter()
            await semaphore.acquire()
            queue_wait_ms = (time.perf_counter() - queue_t0) * 1000.0
        finally:
            async with self._waiters_lock:
                self._waiters = max(0, self._waiters - 1)

        try:
            reset_peak_memory_stats()
            cold_start = self._cold_start_pending

            async def _infer() -> SpeechResult:
                nonlocal inference_ms
                inf_t0 = time.perf_counter()
                result = await run_in_threadpool(self._manager.provider.generate, request)
                inference_ms = (time.perf_counter() - inf_t0) * 1000.0
                return result

            try:
                result = await asyncio.wait_for(
                    _infer(),
                    timeout=settings.REQUEST_TIMEOUT,
                )
            except asyncio.TimeoutError as exc:
                raise RequestTimeoutError(
                    f"Request exceeded REQUEST_TIMEOUT={settings.REQUEST_TIMEOUT}s"
                ) from exc

            if self._cold_start_pending:
                self._cold_start_pending = False

            total_latency_ms = (time.perf_counter() - total_t0) * 1000.0
            mem = gpu_memory_mb()
            log_metrics(
                metrics_logger,
                request_id=request.request_id,
                queue_wait_ms=round(queue_wait_ms, 2),
                inference_ms=round(inference_ms, 2),
                total_latency_ms=round(total_latency_ms, 2),
                characters=len(request.text),
                audio_duration=round(result.duration_seconds, 4),
                gpu_memory_mb=mem["gpu_memory_mb"],
                peak_gpu_memory_mb=mem["peak_gpu_memory_mb"],
                cold_start=cold_start,
                provider=settings.SPEECH_PROVIDER,
                mode=request.mode,
            )
            return result
        finally:
            semaphore.release()


_engine: SpeechEngine | None = None


def get_speech_engine() -> SpeechEngine:
    global _engine
    if _engine is None:
        _engine = SpeechEngine()
    return _engine
