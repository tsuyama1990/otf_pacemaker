"""MaxGamma Sampler Strategy."""

import logging
import numpy as np
from typing import List, Tuple
from ase import Atoms

from src.core.interfaces import Sampler

logger = logging.getLogger(__name__)

class MaxGammaSampler(Sampler):
    """Selects atoms with the highest uncertainty (gamma) values.

    This sampler is typically used when a single frame (Atoms object) is available
    and we simply want to pick the atoms with the highest extrapolation grade.
    """

    def sample(self, **kwargs) -> List[Tuple[Atoms, int]]:
        """Select atom indices with the maximum gamma values from a single Atoms object.

        Args:
            **kwargs: Must contain 'atoms' (Atoms) and 'n_clusters' (int).

        Returns:
            List[Tuple[Atoms, int]]: A list of tuples (atoms, index).
        """
        atoms = kwargs.get('atoms')
        n_clusters = kwargs.get('n_clusters')

        if not atoms:
            raise ValueError("MaxGammaSampler requires 'atoms' in kwargs.")
        if n_clusters is None:
            n_clusters = 1

        # Try to find the gamma array
        # LAMMPS dump usually outputs 'f_f_gamma' if we used 'dump ... f_f_gamma'
        gamma_key = None
        possible_keys = ["f_f_gamma", "f_gamma", "gamma"]

        for key in possible_keys:
            if key in atoms.arrays:
                gamma_key = key
                break

        if not gamma_key:
            available_keys = list(atoms.arrays.keys())
            raise ValueError(f"Gamma values not found in atoms.arrays. Available keys: {available_keys}. "
                             "Ensure the dump file contains the gamma calculation output.")

        gammas = atoms.get_array(gamma_key)
        if gammas.ndim > 1:
            gammas = gammas.flatten()

        # Get indices of top n_clusters gammas
        # np.argsort returns ascending, so we take the last n
        sorted_indices = np.argsort(gammas)
        # Take top n_clusters, reversing to get descending order (highest gamma first)
        top_indices = sorted_indices[-n_clusters:][::-1]

        logger.info(f"Selected {len(top_indices)} atoms. Max gamma: {gammas[top_indices[0]]:.4f}")

        return [(atoms, int(idx)) for idx in top_indices]
