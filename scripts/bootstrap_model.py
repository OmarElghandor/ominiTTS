#!/usr/bin/env python3
"""One-time bootstrap: download OmniVoice weights into MODEL_STORE_DIR.

Run manually against the persistent volume — never invoked by the API entrypoint.

  railway run python scripts/bootstrap_model.py
  MODEL_STORE_DIR=./model-store python scripts/bootstrap_model.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap requires network access even if the shell inherits production offline flags.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from huggingface_hub import snapshot_download  # noqa: E402

MODEL_NAME = os.environ.get("MODEL_NAME", "k2-fsa/OmniVoice")
MODEL_STORE_DIR = Path(os.environ.get("MODEL_STORE_DIR", "/data/omnivoice-model"))
AUDIO_TOKENIZER_REPO = "eustlb/higgs-audio-v2-tokenizer"
AUDIO_TOKENIZER_DIR = MODEL_STORE_DIR / "audio_tokenizer"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def _download_kwargs() -> dict:
    kwargs: dict = {"local_dir_use_symlinks": False}
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def main() -> int:
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    resolved = MODEL_STORE_DIR.resolve()

    print(f"Downloading {MODEL_NAME} into {resolved} ...")
    if endpoint := os.environ.get("HF_ENDPOINT"):
        print(f"Using HF_ENDPOINT mirror: {endpoint}")

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=str(resolved),
        **_download_kwargs(),
    )

    if not AUDIO_TOKENIZER_DIR.is_dir():
        print(f"audio_tokenizer/ missing — downloading {AUDIO_TOKENIZER_REPO} ...")
        snapshot_download(
            repo_id=AUDIO_TOKENIZER_REPO,
            local_dir=str(AUDIO_TOKENIZER_DIR.resolve()),
            **_download_kwargs(),
        )
    else:
        print(f"audio_tokenizer/ already present at {AUDIO_TOKENIZER_DIR.resolve()}")

    total = _dir_size(resolved)
    print(
        f"Bootstrap complete.\n"
        f"  MODEL_STORE_DIR: {resolved}\n"
        f"  Total size:      {_format_size(total)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
