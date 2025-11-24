"""Random Sampler Strategy.

This module implements a simple random sampling strategy.
"""

import logging
import random
from typing import List, Tuple, Any
from ase import Atoms

from src.core.interfaces import Sampler

logger = logging.getLogger(__name__)

class RandomSampler(Sampler):
    """Selects candidates randomly from the pool."""

    def sample(self, candidates: List[Atoms], n_samples: int, **kwargs: Any) -> List[Tuple[Atoms, int]]:
        """Select n_samples randomly from candidates.

        Args:
            candidates: List of candidate structures.
            n_samples: Number of structures to select.
            **kwargs: Additional arguments (ignored).

        Returns:
            List[Tuple[Atoms, int]]: List of selected (structure, original_index) tuples.
        """
        n_candidates = len(candidates)
        if n_candidates == 0:
            return []

        if n_candidates <= n_samples:
            # Return all if fewer than requested
            indices = list(range(n_candidates))
        else:
            indices = random.sample(range(n_candidates), n_samples)

        # Sort indices to maintain some order if needed, though not strictly required
        indices.sort()

        logger.info(f"RandomSampler: Selected {len(indices)} structures from {n_candidates} candidates.")
        return [(candidates[i], i) for i in indices]
