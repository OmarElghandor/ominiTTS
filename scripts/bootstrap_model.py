#!/usr/bin/env python3
"""Bootstrap VoiceTut-TTS weights into MODEL_STORE_DIR.

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
    DEFAULT_MODEL_NAME,
    dir_size_bytes,
    format_size,
    has_hf_cache_layout,
    list_top_level_entries,
    resolve_model_store_dir,
    verify_model_store,
    write_bootstrap_marker,
)

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
AUDIO_TOKENIZER_REPO = "eustlb/higgs-audio-v2-tokenizer"

# VoiceTut HF repo includes ~19 GB of training state — never download those.
IGNORE_PATTERNS = [
    "optimizer.bin",
    "random_states_*",
    "scheduler.bin",
    "train_config.json",
]


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
        f"  MODEL_NAME:                   {MODEL_NAME}\n"
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
    write_bootstrap_marker(store_dir, MODEL_NAME)
    _log_store_context(store_dir, "Bootstrap complete.")


def _fail_verification(store_dir: Path, context: str) -> int:
    errors = verify_model_store(store_dir, model_name=MODEL_NAME)
    print(f"ERROR: {context}", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    _log_store_context(store_dir, "Store state after failed verification:")
    return 1


def _enable_hf_online() -> dict[str, str | None]:
    """Allow Hugging Face downloads for this bootstrap call.

    huggingface_hub caches HF_HUB_OFFLINE at import time, so flipping the env
    alone is not enough if the library was already imported while offline=1.
    """
    prev = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    # XET transfers can stall on some networks; classic HTTP is more reliable for bootstrap.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = False
    except Exception:
        pass
    return prev


def _restore_hf_env(prev: dict[str, str | None]) -> None:
    for key, value in prev.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "") == "1"
    except Exception:
        pass


def run_bootstrap(store_dir: Path | None = None) -> None:
    """Download weights into store_dir. Raises on failure. Safe to call if already valid."""
    prev = _enable_hf_online()

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
        elif verify_model_store(target, model_name=MODEL_NAME):
            if any(target.iterdir()):
                _clear_store_contents(
                    target,
                    "incomplete, mismatched, or invalid model store (verification failed)",
                )
        else:
            print("Valid model store already present — skipping download.")
            _print_success_summary(target)
            return

        print(f"Downloading {MODEL_NAME} into {target} ...")
        print(f"  Ignoring training artifacts: {', '.join(IGNORE_PATTERNS)}")
        print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')!r} (online bootstrap)")
        if endpoint := os.environ.get("HF_ENDPOINT"):
            print(f"Using HF_ENDPOINT mirror: {endpoint}")

        dl_kwargs = {**_download_kwargs(), "local_files_only": False}
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(target),
            ignore_patterns=IGNORE_PATTERNS,
            **dl_kwargs,
        )

        audio_tokenizer_dir = target / "audio_tokenizer"
        if not audio_tokenizer_dir.is_dir() or not (
            audio_tokenizer_dir / "model.safetensors"
        ).is_file():
            print(f"audio_tokenizer/ missing — downloading {AUDIO_TOKENIZER_REPO} ...")
            snapshot_download(
                repo_id=AUDIO_TOKENIZER_REPO,
                local_dir=str(audio_tokenizer_dir),
                **dl_kwargs,
            )

        _log_store_context(target, "Post-download store state:")
        errors = verify_model_store(target, model_name=MODEL_NAME)
        if errors:
            # Marker may be missing until success write — filter mismatch-on-missing-marker
            # after files are present by writing then re-checking isn't needed; only
            # structural errors should fail here. Marker mismatch only applies when a
            # marker already exists for a different model.
            structural = [
                e
                for e in errors
                if not e.startswith("model mismatch:")
            ]
            if structural:
                detail = "; ".join(structural)
                raise RuntimeError(f"post-download verification failed: {detail}")

        _print_success_summary(target)
    finally:
        _restore_hf_env(prev)


def main() -> int:
    try:
        run_bootstrap()
        return 0
    except Exception as exc:
        print(f"ERROR: bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
