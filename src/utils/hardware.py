"""Hardware utility functions."""

import logging
import torch
import math

logger = logging.getLogger(__name__)

def get_available_vram() -> int:
    """Get total available VRAM in bytes on the primary CUDA device.

    Returns:
        int: Total VRAM in bytes. Returns 0 if no CUDA device is available.
    """
    if not torch.cuda.is_available():
        return 0

    try:
        # Get free and total memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        return total_mem
    except Exception as e:
        logger.warning(f"Failed to get VRAM info: {e}")
        return 0

def suggest_batch_size(vram_bytes: int) -> int:
    """Suggest a batch size based on available VRAM.

    Heuristic:
    - < 4GB: 16
    - 4GB - 8GB: 32
    - 8GB - 16GB: 64
    - 16GB - 24GB: 128
    - > 24GB: 256

    Args:
        vram_bytes: VRAM in bytes.

    Returns:
        int: Recommended batch size.
    """
    if vram_bytes <= 0:
        return 32

    gb = vram_bytes / (1024**3)

    if gb < 4:
        return 16
    elif gb < 8:
        return 32
    elif gb < 16:
        return 64
    elif gb < 24:
        return 128
    else:
        return 256
