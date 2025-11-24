"""Max Gamma Sampler Strategy.

This module implements a sampling strategy that selects structures with the
highest extrapolation grade (gamma), indicating high uncertainty.
"""

import logging
from typing import List, Tuple, Any
import numpy as np
from ase import Atoms

from src.core.interfaces import Sampler

logger = logging.getLogger(__name__)

class MaxGammaSampler(Sampler):
    """Selects candidates with the highest uncertainty (Max Gamma)."""

    def sample(self, candidates: List[Atoms], n_samples: int, **kwargs: Any) -> List[Tuple[Atoms, int]]:
        """Select n_samples with highest max gamma.

        Args:
            candidates: List of candidate structures.
            n_samples: Number of structures to select.
            **kwargs: Must contain 'gammas' (List[float]) corresponding to candidates,
                      or we assume gammas are stored in atoms.info['max_gamma'].

        Returns:
            List[Tuple[Atoms, int]]: List of selected (structure, original_index) tuples.
        """
        n_candidates = len(candidates)
        if n_candidates == 0:
            return []

        if n_samples >= n_candidates:
             return [(c, i) for i, c in enumerate(candidates)]

        # Extract gammas
        # We expect 'gammas' in kwargs OR 'max_gamma' in atoms.info
        gammas = kwargs.get('gammas')

        if gammas is None:
            # Try to get from info
            try:
                gammas = [atoms.info.get('max_gamma', 0.0) for atoms in candidates]
            except Exception as e:
                logger.warning(f"Could not retrieve gammas from atoms.info: {e}. defaulting to 0.0")
                gammas = [0.0] * n_candidates

        if len(gammas) != n_candidates:
            logger.error("Length of gammas does not match candidates.")
            return []

        # Sort by gamma descending
        # valid indices
        indices = list(range(n_candidates))

        # Sort indices based on gamma values (descending)
        indices.sort(key=lambda i: gammas[i], reverse=True)

        selected_indices = indices[:n_samples]

        # Log selection
        logger.info(f"MaxGammaSampler: Selected {len(selected_indices)} structures. "
                    f"Top Gamma: {gammas[selected_indices[0]]:.4f}, "
                    f"Bottom Gamma: {gammas[selected_indices[-1]]:.4f}")

        return [(candidates[i], i) for i in selected_indices]
