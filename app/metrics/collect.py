"""GPU / timing metric helpers."""

from __future__ import annotations

from typing import Any

import torch


def gpu_memory_mb() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "gpu_memory_mb": None,
            "peak_gpu_memory_mb": None,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
        }
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    return {
        "gpu_memory_mb": round(allocated, 2),
        "peak_gpu_memory_mb": round(peak, 2),
        "gpu_memory_allocated_mb": round(allocated, 2),
        "gpu_memory_reserved_mb": round(reserved, 2),
    }


def reset_peak_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def status_gpu_fields() -> dict[str, Any]:
    mem = gpu_memory_mb()
    return {
        "gpu_memory_allocated_mb": mem["gpu_memory_allocated_mb"],
        "gpu_memory_reserved_mb": mem["gpu_memory_reserved_mb"],
    }
