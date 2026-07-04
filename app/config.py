import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.model_store import assert_model_store_verified, resolve_model_store_dir

logger = logging.getLogger(__name__)

MODEL_STORE_EMPTY_MSG = (
    "MODEL_STORE_DIR is empty — set BOOTSTRAP_ONLY=1 in Railway, deploy, wait for "
    "'Bootstrap complete' (~3 GB), remove the variable, then redeploy"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MODEL_NAME: str = "k2-fsa/OmniVoice"
    MODEL_STORE_DIR: str = "/data/omnivoice-model"
    DEVICE: str = "cuda:0"
    DTYPE: str | None = None
    MAX_TEXT_LENGTH: int = 500
    PORT: int = 8080

    def resolve_model_path(self) -> Path:
        return Path(self.MODEL_STORE_DIR).resolve()

    def model_store_has_content(self) -> bool:
        store = self.resolve_model_path()
        if not store.is_dir():
            return False
        return any(store.rglob("*"))

    def assert_model_store_ready(self) -> None:
        store = self.resolve_model_path()
        if not store.is_dir() or not self.model_store_has_content():
            raise RuntimeError(MODEL_STORE_EMPTY_MSG)
        assert_model_store_verified(store)

    def assert_offline_mode(self) -> None:
        hf_offline = os.environ.get("HF_HUB_OFFLINE", "")
        transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE", "")
        if hf_offline != "1" or transformers_offline != "1":
            raise RuntimeError(
                "HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must both be set to 1 in production. "
                f"Got HF_HUB_OFFLINE={hf_offline!r} TRANSFORMERS_OFFLINE={transformers_offline!r}"
            )
        logger.info(
            "Offline mode active: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 — "
            "API will not contact huggingface.co"
        )

    def resolve_device_and_dtype(self) -> tuple[str, torch.dtype]:
        requested_device = self.DEVICE.strip()

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA not available — falling back to CPU. Inference will be significantly slower."
            )
            resolved_device = "cpu"
        else:
            resolved_device = requested_device

        if self.DTYPE:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            resolved_dtype = dtype_map.get(self.DTYPE.lower())
            if resolved_dtype is None:
                raise ValueError(f"Unsupported DTYPE: {self.DTYPE}")
        elif resolved_device == "cpu":
            resolved_dtype = torch.float32
        else:
            resolved_dtype = torch.float16

        return resolved_device, resolved_dtype


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    resolved = str(resolve_model_store_dir())
    settings.MODEL_STORE_DIR = resolved
    logger.info(
        "MODEL_STORE_DIR=%s RAILWAY_VOLUME_MOUNT_PATH=%s",
        resolved,
        os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "<unset>"),
    )
    return settings
