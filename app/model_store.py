"""Shared MODEL_STORE_DIR layout checks for bootstrap and runtime."""

from __future__ import annotations

import os
from pathlib import Path

BOOTSTRAP_COMPLETE_MARKER = ".omnivoice-bootstrap-complete"

REQUIRED_RELATIVE_PATHS = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
)

MIN_FILE_BYTES: dict[str, int] = {
    "model.safetensors": 2 * 1024**3,
    "audio_tokenizer/model.safetensors": 750 * 1024**2,
}

BOOTSTRAP_RERUN_MSG = (
    "MODEL_STORE_DIR failed verification — re-run scripts/bootstrap_model.py against the "
    "volume and wait for 'Bootstrap complete' (~3 GB)"
)


def resolve_model_store_dir() -> Path:
    """Resolve the model store path (must match the container volume mount)."""
    raw = os.environ.get("MODEL_STORE_DIR") or "/data/omnivoice-model"
    return Path(raw).resolve()


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def verify_model_store(store_dir: Path) -> list[str]:
    """Return human-readable errors; empty list means the store is usable."""
    errors: list[str] = []
    for rel in REQUIRED_RELATIVE_PATHS:
        path = store_dir / rel
        if not path.is_file():
            errors.append(f"missing: {rel}")
            continue
        if min_bytes := MIN_FILE_BYTES.get(rel):
            size = path.stat().st_size
            if size < min_bytes:
                errors.append(
                    f"undersized: {rel} ({format_size(size)}, need >= {format_size(min_bytes)})"
                )
    return errors


def has_hf_cache_layout(store_dir: Path) -> bool:
    """True when a prior cache_dir-style download left models--* subfolders."""
    if not store_dir.is_dir():
        return False
    return any(
        child.is_dir() and child.name.startswith("models--") for child in store_dir.iterdir()
    )


def assert_model_store_verified(store_dir: Path) -> None:
    errors = verify_model_store(store_dir)
    if errors:
        detail = "; ".join(errors)
        raise RuntimeError(f"{BOOTSTRAP_RERUN_MSG} ({detail})")


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def list_top_level_entries(store_dir: Path) -> list[str]:
    if not store_dir.is_dir():
        return []
    return sorted(
        f"{entry.name}/" if entry.is_dir() else entry.name for entry in store_dir.iterdir()
    )


def write_bootstrap_marker(store_dir: Path) -> None:
    (store_dir / BOOTSTRAP_COMPLETE_MARKER).write_text(
        f"verified ok at {store_dir.resolve()}\n",
        encoding="utf-8",
    )


def has_bootstrap_marker(store_dir: Path) -> bool:
    return (store_dir / BOOTSTRAP_COMPLETE_MARKER).is_file()
