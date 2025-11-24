"""Composite Sampler Strategy.

This module implements a composite sampling strategy that combines
uncertainty-based sampling (MaxGamma) with random sampling to ensure
both exploration and exploitation (diversity).
"""

import logging
from typing import List, Tuple, Any
from ase import Atoms

from src.core.interfaces import Sampler
from src.sampling.strategies.max_gamma import MaxGammaSampler
from src.sampling.strategies.random_sampler import RandomSampler

logger = logging.getLogger(__name__)

class CompositeSampler(Sampler):
    """Combines MaxGamma and Random sampling."""

    def __init__(self, gamma_sampler: MaxGammaSampler = None, random_sampler: RandomSampler = None):
        """Initialize the CompositeSampler.

        Args:
            gamma_sampler: Instance of MaxGammaSampler (optional, defaults to new instance).
            random_sampler: Instance of RandomSampler (optional, defaults to new instance).
        """
        self.gamma_sampler = gamma_sampler or MaxGammaSampler()
        self.random_sampler = random_sampler or RandomSampler()

    def sample(self, candidates: List[Atoms], n_samples: int, **kwargs: Any) -> List[Tuple[Atoms, int]]:
        """Select n_samples using a mix of strategies.

        Logic:
        - If N=1: 100% MaxGamma.
        - If N>=2: At least 1 Random, rest MaxGamma (target ratio 80/20).

        Args:
            candidates: List of candidate structures.
            n_samples: Total number of structures to select.
            **kwargs: Additional arguments passed to samplers (e.g. gamma_threshold).

        Returns:
            List[Tuple[Atoms, int]]: Combined list of selected (structure, index) tuples.
        """
        n_candidates = len(candidates)
        if n_candidates == 0:
            return []

        # If requested samples exceed candidates, return all
        if n_samples >= n_candidates:
             return [(c, i) for i, c in enumerate(candidates)]

        # Determine split
        if n_samples == 1:
            n_random = 0
            n_gamma = 1
        else:
            # Target 20% random
            target_random = int(n_samples * 0.2)
            # Ensure at least 1 random if n_samples >= 2
            n_random = max(1, target_random)
            n_gamma = n_samples - n_random

        logger.info(f"CompositeSampler: Splitting {n_samples} samples -> {n_gamma} MaxGamma + {n_random} Random.")

        selected_indices = set()
        results = []

        # 1. Max Gamma Sampling
        if n_gamma > 0:
            gamma_results = self.gamma_sampler.sample(candidates, n_samples=n_gamma, **kwargs)
            for atom, idx in gamma_results:
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    results.append((atom, idx))

        # 2. Random Sampling
        # We need to sample from the *remaining* pool or just sample and check duplicates.
        # RandomSampler samples from the provided list.
        # If we pass the full list, we might pick duplicates.
        # Efficient approach: Pass full list, ask for more than needed if we expect collisions,
        # or just handle the logic here manually since RandomSampler is simple.
        # Using the RandomSampler instance for consistency.

        if n_random > 0:
            # We filter out already selected to avoid duplicates
            remaining_candidates = []
            original_indices_map = []

            for i, c in enumerate(candidates):
                if i not in selected_indices:
                    remaining_candidates.append(c)
                    original_indices_map.append(i)

            if len(remaining_candidates) < n_random:
                # Take all remaining
                for k, c in enumerate(remaining_candidates):
                    orig_idx = original_indices_map[k]
                    selected_indices.add(orig_idx)
                    results.append((c, orig_idx))
            else:
                random_results = self.random_sampler.sample(remaining_candidates, n_samples=n_random, **kwargs)
                for atom, local_idx in random_results:
                    # local_idx is index in remaining_candidates
                    orig_idx = original_indices_map[local_idx]
                    if orig_idx not in selected_indices:
                         selected_indices.add(orig_idx)
                         results.append((atom, orig_idx))

        # Sort by index to maintain deterministic order if helpful
        results.sort(key=lambda x: x[1])

        return results
