"""Shared MODEL_STORE_DIR layout checks for bootstrap and runtime."""

from __future__ import annotations

import os
from pathlib import Path

BOOTSTRAP_COMPLETE_MARKER = ".omnivoice-bootstrap-complete"
DEFAULT_MODEL_NAME = "mohammedaly22/VoiceTut-TTS"

REQUIRED_RELATIVE_PATHS = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
    "reference_speakers/references.json",
)

MIN_FILE_BYTES: dict[str, int] = {
    "model.safetensors": 2 * 1024**3,
    "audio_tokenizer/model.safetensors": 750 * 1024**2,
}

BOOTSTRAP_RERUN_MSG = (
    "MODEL_STORE_DIR failed verification — re-run scripts/bootstrap_model.py against the "
    "volume and wait for 'Bootstrap complete' (~3.5 GB)"
)

REFERENCE_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def resolve_model_store_dir() -> Path:
    """Resolve the model store path (must match the container volume mount).

    On RunPod Serverless the Network Volume is at /runpod-volume. Prefer that
    whenever the mount exists, even if MODEL_STORE_DIR was left at the Pod
    default (/data/omnivoice-model).
    """
    raw = os.environ.get("MODEL_STORE_DIR")
    runpod_store = Path("/runpod-volume/omnivoice-model")
    pod_default = Path("/data/omnivoice-model")

    if Path("/runpod-volume").is_dir():
        # Volume attached (Serverless). Ignore mistaken Pod-default env.
        if raw:
            configured = Path(raw)
            if configured.resolve() != pod_default.resolve():
                return configured.resolve()
        return runpod_store.resolve()

    if raw:
        return Path(raw).resolve()
    return pod_default.resolve()


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def expected_model_name() -> str:
    return os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME


def read_bootstrap_marker_repo(store_dir: Path) -> str | None:
    marker = store_dir / BOOTSTRAP_COMPLETE_MARKER
    if not marker.is_file():
        return None
    for line in marker.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("model_name="):
            return line.split("=", 1)[1].strip() or None
    return None


def verify_model_store(store_dir: Path, *, model_name: str | None = None) -> list[str]:
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

    refs_dir = store_dir / "reference_speakers"
    if refs_dir.is_dir():
        has_audio = any(
            p.is_file() and p.suffix.lower() in REFERENCE_AUDIO_SUFFIXES
            for p in refs_dir.iterdir()
        )
        if not has_audio:
            errors.append("missing: reference_speakers/* audio file")
    else:
        # Already reported via references.json missing, but keep explicit.
        if "missing: reference_speakers/references.json" not in errors:
            errors.append("missing: reference_speakers/")

    expected = model_name or expected_model_name()
    marker_repo = read_bootstrap_marker_repo(store_dir)
    if marker_repo is not None and marker_repo != expected:
        errors.append(
            f"model mismatch: store marked as {marker_repo!r}, expected {expected!r}"
        )
    elif marker_repo is None and (store_dir / BOOTSTRAP_COMPLETE_MARKER).is_file():
        # Legacy marker from base OmniVoice bootstrap — force re-seed for VoiceTut.
        errors.append(
            f"model mismatch: legacy bootstrap marker (no model_name); expected {expected!r}"
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


def write_bootstrap_marker(store_dir: Path, model_name: str | None = None) -> None:
    repo = model_name or expected_model_name()
    (store_dir / BOOTSTRAP_COMPLETE_MARKER).write_text(
        f"verified ok at {store_dir.resolve()}\nmodel_name={repo}\n",
        encoding="utf-8",
    )


def has_bootstrap_marker(store_dir: Path) -> bool:
    return (store_dir / BOOTSTRAP_COMPLETE_MARKER).is_file()
