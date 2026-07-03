#!/usr/bin/env python3
"""One-time bootstrap: download OmniVoice weights into MODEL_STORE_DIR.

Run manually against the persistent volume — never invoked by the API entrypoint.

  railway run python scripts/bootstrap_model.py
  MODEL_STORE_DIR=./model-store python scripts/bootstrap_model.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

# Bootstrap requires network access even if the shell inherits production offline flags.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from huggingface_hub import snapshot_download  # noqa: E402

from app.model_store import (  # noqa: E402
    dir_size_bytes,
    format_size,
    has_hf_cache_layout,
    list_top_level_entries,
    verify_model_store,
)

MODEL_NAME = os.environ.get("MODEL_NAME", "k2-fsa/OmniVoice")
MODEL_STORE_DIR = Path(
    os.environ.get("MODEL_STORE_DIR")
    or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    or "/data/omnivoice-model"
)
AUDIO_TOKENIZER_REPO = "eustlb/higgs-audio-v2-tokenizer"
AUDIO_TOKENIZER_DIR = MODEL_STORE_DIR / "audio_tokenizer"


def _download_kwargs() -> dict:
    kwargs: dict = {"local_dir_use_symlinks": False}
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def _clear_store_contents(store_dir: Path, reason: str) -> None:
    if not store_dir.is_dir():
        return
    entries = sorted(store_dir.iterdir(), key=lambda p: p.name)
    if not entries:
        return
    print(f"Clearing {store_dir.resolve()} ({reason}):")
    for entry in entries:
        print(f"  removing {entry.name}")
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _print_success_summary(store_dir: Path) -> None:
    total = dir_size_bytes(store_dir)
    top_level = list_top_level_entries(store_dir)
    print(
        "Bootstrap complete.\n"
        f"  MODEL_STORE_DIR: {store_dir.resolve()}\n"
        f"  Total size:      {format_size(total)}\n"
        f"  Top-level:       {', '.join(top_level)}"
    )


def _fail_verification(store_dir: Path, context: str) -> int:
    errors = verify_model_store(store_dir)
    print(f"ERROR: {context}", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def main() -> int:
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    resolved = MODEL_STORE_DIR.resolve()

    if has_hf_cache_layout(resolved):
        _clear_store_contents(
            resolved,
            "HF cache_dir-style layout detected (models--* subfolders)",
        )
    elif verify_model_store(resolved):
        if any(resolved.iterdir()):
            _clear_store_contents(
                resolved,
                "incomplete or invalid model store (verification failed)",
            )
    else:
        print(f"Valid model store already present at {resolved} — skipping download.")
        _print_success_summary(resolved)
        return 0

    print(f"Downloading {MODEL_NAME} into {resolved} ...")
    if endpoint := os.environ.get("HF_ENDPOINT"):
        print(f"Using HF_ENDPOINT mirror: {endpoint}")

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=str(resolved),
        **_download_kwargs(),
    )

    if not AUDIO_TOKENIZER_DIR.is_dir() or not (AUDIO_TOKENIZER_DIR / "model.safetensors").is_file():
        print(f"audio_tokenizer/ missing — downloading {AUDIO_TOKENIZER_REPO} ...")
        snapshot_download(
            repo_id=AUDIO_TOKENIZER_REPO,
            local_dir=str(AUDIO_TOKENIZER_DIR.resolve()),
            **_download_kwargs(),
        )

    errors = verify_model_store(resolved)
    if errors:
        return _fail_verification(resolved, "post-download verification failed")

    _print_success_summary(resolved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
