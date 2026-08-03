"""ModelManager — singleton CUDA init, load, warmup, and readiness status."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import torch

from app.config import Settings
from app.metrics.collect import status_gpu_fields
from app.model_store import verify_model_store
from app.speech.provider import SpeechProvider
from app.speech.providers.omnivoice import create_provider

logger = logging.getLogger(__name__)


def _acquire_bootstrap_lock(store: Path):
    """Exclusive lock so only one worker seeds the shared Network Volume."""
    store.mkdir(parents=True, exist_ok=True)
    lock_path = store / ".bootstrap.lock"
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    except Exception:
        pass
    return lock_file


def _release_bootstrap_lock(lock_file) -> None:
    try:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock_file.close()
    except Exception:
        pass


class ModelManager:
    """Loads the speech provider exactly once; never reloads between requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._provider: SpeechProvider | None = None
        self._cuda_initialized = False
        self._model_loaded = False
        self._tokenizer_loaded = False
        self._warmup_completed = False
        self._loading = False
        self._load_error: str | None = None
        self._device: str = "cpu"

    @property
    def provider(self) -> SpeechProvider:
        if self._provider is None or not self._model_loaded:
            raise RuntimeError("Model is not loaded")
        return self._provider

    def is_ready(self) -> bool:
        return (
            self._model_loaded
            and self._tokenizer_loaded
            and self._warmup_completed
            and self._load_error is None
        )

    def is_loading(self) -> bool:
        return self._loading

    def get_load_error(self) -> str | None:
        return self._load_error

    def get_device(self) -> str:
        return self._device

    def status(self) -> dict[str, Any]:
        return {
            "cuda_initialized": self._cuda_initialized,
            "model_loaded": self._model_loaded,
            "tokenizer_loaded": self._tokenizer_loaded,
            "warmup_completed": self._warmup_completed,
            "loading": self._loading,
            "load_error": self._load_error,
            "device": self._device,
            "provider": getattr(self._provider, "name", None) if self._provider else None,
            "ready": self.is_ready(),
            **status_gpu_fields(),
        }

    def initialize_cuda(self, settings: Settings) -> None:
        resolved_device, _ = settings.resolve_device_and_dtype()
        self._device = resolved_device
        if resolved_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "DEVICE requests CUDA but torch.cuda.is_available() is False"
                )
            idx = 0
            if ":" in resolved_device:
                try:
                    idx = int(resolved_device.split(":", 1)[1])
                except ValueError:
                    idx = 0
            torch.cuda.set_device(idx)
            _ = torch.zeros(1, device=resolved_device)
            self._cuda_initialized = True
            logger.info("CUDA initialized on %s", resolved_device)
        else:
            self._cuda_initialized = False
            logger.info("Running on CPU (DEVICE=%s)", resolved_device)

    def _store_needs_bootstrap(self, settings: Settings) -> bool:
        store = settings.resolve_model_path()
        if not store.is_dir() or not any(store.iterdir()):
            return True
        return bool(verify_model_store(store))

    def _bootstrap_if_empty(self, settings: Settings) -> None:
        store = settings.resolve_model_path()
        logger.warning(
            "BOOTSTRAP_IF_EMPTY=1 and store is empty/invalid at %s — "
            "downloading ~3 GB (this cold start will be slow)",
            store,
        )
        import os
        import subprocess
        import sys

        bootstrap_path = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_model.py"
        lock_file = _acquire_bootstrap_lock(store)
        try:
            if not self._store_needs_bootstrap(settings):
                logger.info("Store already seeded by another worker — skipping download")
                return
            # Fresh process so huggingface_hub does not keep a cached HF_HUB_OFFLINE=1.
            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "0"
            env["TRANSFORMERS_OFFLINE"] = "0"
            env["MODEL_STORE_DIR"] = str(store)
            logger.info("Running bootstrap subprocess: %s", bootstrap_path)
            subprocess.run(
                [sys.executable, str(bootstrap_path)],
                env=env,
                check=True,
            )
        finally:
            _release_bootstrap_lock(lock_file)

    def load(self, settings: Settings) -> None:
        """Load provider once. Raises on failure. Downloads only if BOOTSTRAP_IF_EMPTY=1."""
        with self._lock:
            if self.is_ready():
                return
            if self._model_loaded and self._warmup_completed:
                return

            self._loading = True
            self._load_error = None
            try:
                if self._store_needs_bootstrap(settings) and settings.BOOTSTRAP_IF_EMPTY:
                    self._bootstrap_if_empty(settings)

                settings.assert_offline_mode()
                self.initialize_cuda(settings)

                store = settings.resolve_model_path()
                logger.info(
                    "Loading speech provider from MODEL_STORE_DIR=%s (exists=%s)",
                    store,
                    store.is_dir() and any(store.iterdir()) if store.is_dir() else False,
                )

                provider = create_provider(settings.SPEECH_PROVIDER)
                provider.load(settings)
                self._provider = provider
                self._model_loaded = provider.is_loaded()
                self._tokenizer_loaded = provider.tokenizer_loaded()
                self._device = getattr(provider, "device", self._device)

                if not self._model_loaded:
                    raise RuntimeError("Provider reported model not loaded after load()")

                logger.info("Running warmup inference: %r", settings.WARMUP_TEXT)
                provider.warmup(settings.WARMUP_TEXT)
                self._warmup_completed = True
                self._loading = False
                logger.info("ModelManager ready: %s", self.status())
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._load_error = message
                self._loading = False
                self._model_loaded = False
                self._tokenizer_loaded = False
                self._warmup_completed = False
                logger.error("ModelManager load failed: %s", message)
                raise

    def clear_cuda_cache(self) -> None:
        if self._provider is not None:
            self._provider.clear_cuda_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


_manager: ModelManager | None = None
_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ModelManager()
        return _manager
