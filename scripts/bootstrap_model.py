#!/usr/bin/env python3
"""Bootstrap OmniVoice weights into MODEL_STORE_DIR.

Manual:
  docker compose run --rm -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \\
    omnivoice-api python scripts/bootstrap_model.py
  MODEL_STORE_DIR=./model-store python scripts/bootstrap_model.py

Automatic (opt-in): set BOOTSTRAP_IF_EMPTY=1 on the Serverless endpoint so the
first worker seeds an empty Network Volume, then loads offline.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from app.model_store import (  # noqa: E402
    dir_size_bytes,
    format_size,
    has_hf_cache_layout,
    list_top_level_entries,
    resolve_model_store_dir,
    verify_model_store,
    write_bootstrap_marker,
)

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "k2-fsa/OmniVoice")
AUDIO_TOKENIZER_REPO = "eustlb/higgs-audio-v2-tokenizer"


def _download_kwargs() -> dict:
    kwargs: dict = {"local_dir_use_symlinks": False}
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def _log_store_context(store_dir: Path, label: str) -> None:
    top_level = list_top_level_entries(store_dir)
    total = dir_size_bytes(store_dir) if store_dir.is_dir() else 0
    msg = (
        f"{label}\n"
        f"  MODEL_STORE_DIR:              {store_dir}\n"
        f"  On-disk size:                 {format_size(total)}\n"
        f"  Top-level ({len(top_level)}):  {', '.join(top_level) if top_level else '<empty>'}"
    )
    print(msg)
    logger.info(msg.replace("\n", " | "))


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


def run_bootstrap(store_dir: Path | None = None) -> None:
    """Download weights into store_dir. Raises on failure. Safe to call if already valid."""
    # Bootstrap needs HF network even when production env is offline.
    prev_hf = os.environ.get("HF_HUB_OFFLINE")
    prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    try:
        from huggingface_hub import snapshot_download

        target = (store_dir or resolve_model_store_dir()).resolve()
        target.mkdir(parents=True, exist_ok=True)
        _log_store_context(target, "Bootstrap starting.")

        if has_hf_cache_layout(target):
            _clear_store_contents(
                target,
                "HF cache_dir-style layout detected (models--* subfolders)",
            )
        elif verify_model_store(target):
            if any(target.iterdir()):
                _clear_store_contents(
                    target,
                    "incomplete or invalid model store (verification failed)",
                )
        else:
            print("Valid model store already present — skipping download.")
            _print_success_summary(target)
            return

        print(f"Downloading {MODEL_NAME} into {target} ...")
        if endpoint := os.environ.get("HF_ENDPOINT"):
            print(f"Using HF_ENDPOINT mirror: {endpoint}")

        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(target),
            **_download_kwargs(),
        )

        audio_tokenizer_dir = target / "audio_tokenizer"
        if not audio_tokenizer_dir.is_dir() or not (
            audio_tokenizer_dir / "model.safetensors"
        ).is_file():
            print(f"audio_tokenizer/ missing — downloading {AUDIO_TOKENIZER_REPO} ...")
            snapshot_download(
                repo_id=AUDIO_TOKENIZER_REPO,
                local_dir=str(audio_tokenizer_dir),
                **_download_kwargs(),
            )

        _log_store_context(target, "Post-download store state:")
        errors = verify_model_store(target)
        if errors:
            detail = "; ".join(errors)
            raise RuntimeError(f"post-download verification failed: {detail}")

        _print_success_summary(target)
    finally:
        if prev_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_hf
        if prev_tf is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prev_tf


def main() -> int:
    try:
        run_bootstrap()
        return 0
    except Exception as exc:
        print(f"ERROR: bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
