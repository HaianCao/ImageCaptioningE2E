"""Memory cleanup helpers for GPU-heavy notebook and training flows."""

from __future__ import annotations

import gc

import torch


def cleanup_cuda_memory(note: str | None = None, synchronize: bool = True) -> None:
    """Release Python and CUDA cached memory without touching live tensors.

    This is useful after feature extraction or after a training phase when the
    model still needs to stay in memory, but temporary CUDA cache blocks can be
    returned to the system.
    """
    gc.collect()

    if torch.cuda.is_available():
        if synchronize:
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except (AttributeError, RuntimeError):
            pass

    message = "[Cleanup] Released cached memory"
    if note:
        message = f"{message} ({note})"
    print(message)