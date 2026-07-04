#!/usr/bin/env python3
"""One-time bootstrap: download OmniVoice weights into MODEL_STORE_DIR.

Run manually against the persistent volume — never invoked by the API entrypoint.

  railway ssh --service ominiTTS -- python /app/scripts/bootstrap_model.py
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
    resolve_model_store_dir,
    verify_model_store,
    write_bootstrap_marker,
)

MODEL_NAME = os.environ.get("MODEL_NAME", "k2-fsa/OmniVoice")
AUDIO_TOKENIZER_REPO = "eustlb/higgs-audio-v2-tokenizer"


def _download_kwargs() -> dict:
    kwargs: dict = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def _log_store_context(store_dir: Path, label: str) -> None:
    top_level = list_top_level_entries(store_dir)
    total = dir_size_bytes(store_dir) if store_dir.is_dir() else 0
    print(
        f"{label}\n"
        f"  MODEL_STORE_DIR:              {store_dir}\n"
        f"  RAILWAY_VOLUME_MOUNT_PATH:    {os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '<unset>')}\n"
        f"  On-disk size:                 {format_size(total)}\n"
        f"  Top-level ({len(top_level)}):  {', '.join(top_level) if top_level else '<empty>'}"
    )


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
    write_bootstrap_marker(store_dir)
    _log_store_context(store_dir, "Bootstrap complete.")


def _fail_verification(store_dir: Path, context: str) -> int:
    errors = verify_model_store(store_dir)
    print(f"ERROR: {context}", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    _log_store_context(store_dir, "Store state after failed verification:")
    return 1


def main() -> int:
    store_dir = resolve_model_store_dir()
    store_dir.mkdir(parents=True, exist_ok=True)

    _log_store_context(store_dir, "Bootstrap starting.")

    if has_hf_cache_layout(store_dir):
        _clear_store_contents(
            store_dir,
            "HF cache_dir-style layout detected (models--* subfolders)",
        )
    elif verify_model_store(store_dir):
        if any(store_dir.iterdir()):
            _clear_store_contents(
                store_dir,
                "incomplete or invalid model store (verification failed)",
            )
    else:
        print(f"Valid model store already present — skipping download.")
        _print_success_summary(store_dir)
        return 0

    print(f"Downloading {MODEL_NAME} into {store_dir} ...")
    if endpoint := os.environ.get("HF_ENDPOINT"):
        print(f"Using HF_ENDPOINT mirror: {endpoint}")

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=str(store_dir),
        **_download_kwargs(),
    )

    audio_tokenizer_dir = store_dir / "audio_tokenizer"
    if not audio_tokenizer_dir.is_dir() or not (audio_tokenizer_dir / "model.safetensors").is_file():
        print(f"audio_tokenizer/ missing — downloading {AUDIO_TOKENIZER_REPO} ...")
        snapshot_download(
            repo_id=AUDIO_TOKENIZER_REPO,
            local_dir=str(audio_tokenizer_dir),
            **_download_kwargs(),
        )

    _log_store_context(store_dir, "Post-download store state:")

    errors = verify_model_store(store_dir)
    if errors:
        return _fail_verification(store_dir, "post-download verification failed")

    _print_success_summary(store_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
