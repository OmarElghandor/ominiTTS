import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MODEL_NAME: str = "k2-fsa/OmniVoice"
    DEVICE: str = "cuda:0"
    DTYPE: str | None = None
    HF_HOME: str = "/data/huggingface"
    MAX_TEXT_LENGTH: int = 500
    PORT: int = 8080

    def configure_hf_home(self) -> None:
        os.environ["HF_HOME"] = self.HF_HOME
        os.environ.setdefault("HF_HUB_CACHE", str(Path(self.HF_HOME) / "hub"))
        try:
            Path(self.HF_HOME).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create HF_HOME at '{self.HF_HOME}'. "
                "Check volume mount and set RAILWAY_RUN_UID=0 on Railway if needed."
            ) from exc

    def hf_cache_has_content(self) -> bool:
        cache = Path(self.HF_HOME)
        if not cache.is_dir():
            return False
        return any(cache.rglob("*"))

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
    settings.configure_hf_home()
    return settings
