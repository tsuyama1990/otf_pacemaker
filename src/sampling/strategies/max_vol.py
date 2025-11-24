"""MaxVol Sampler Strategy."""

import logging
import numpy as np
from typing import List, Tuple
from ase import Atoms

from src.core.interfaces import Sampler
from src.sampling.strategies.ace_sampler import ACESampler

try:
    import pyace
except ImportError:
    pyace = None

logger = logging.getLogger(__name__)

class MaxVolSampler(Sampler):
    """Selects high-value structures using MaxVol algorithm via pyace/pacemaker API.

    This sampler computes ACE descriptors for a list of structures and uses the
    MaxVol algorithm to select the most informative ones.
    """

    def __init__(self):
        """Initialize MaxVolSampler."""
        pass

    def sample(self, **kwargs) -> List[Tuple[Atoms, int]]:
        """Select structures using MaxVol on ACE descriptors.

        Args:
            **kwargs: Must contain:
                - structures (List[Atoms]): List of candidate structures.
                - potential_path (str): Path to the potential/basis set (YAML or YACE).
                - n_clusters (int): Number of structures to select.

        Returns:
            List[Tuple[Atoms, int]]: List of (structure, max_gamma_atom_index).
        """
        structures = kwargs.get('structures')
        potential_path = kwargs.get('potential_path')
        n_selection = kwargs.get('n_clusters')

        if not structures:
            logger.warning("No structures provided to MaxVolSampler.")
            return []

        if not potential_path:
            raise ValueError("potential_path is required for MaxVolSampler.")

        if not n_selection:
             raise ValueError("n_clusters is required for MaxVolSampler.")

        # 1. Compute Descriptors
        logger.info(f"Computing descriptors for {len(structures)} structures...")
        sampler = ACESampler(potential_path)

        descriptors_list = []
        valid_indices = []

        for i, atoms in enumerate(structures):
            try:
                desc = sampler.compute_descriptors(atoms)

                # Aggregate to structure vector (mean) if needed
                if desc.ndim == 2:
                    desc_vec = np.mean(desc, axis=0)
                else:
                    desc_vec = desc

                descriptors_list.append(desc_vec)
                valid_indices.append(i)

            except Exception as e:
                logger.warning(f"Failed descriptor calc for structure {i}: {e}")

        if not descriptors_list:
            return []

        X = np.array(descriptors_list)

        # 2. MaxVol Selection
        selected_local_indices = []

        if pyace and hasattr(pyace, "SelectMaxVol"):
             # Hypothetical API based on standard PyACE usage patterns
             selected_local_indices = pyace.SelectMaxVol(X, n_selection)
        else:
             # Fallback: QR pivoting
             from scipy.linalg import qr

             limit = min(n_selection, X.shape[1], X.shape[0])
             if limit < n_selection:
                 logger.warning(f"Requested {n_selection} but capped at {limit}.")

             _, _, P = qr(X.T, pivoting=True)
             selected_local_indices = P[:limit]

        # 3. Identify Max Gamma Atom in selected structures
        results = []
        for idx in selected_local_indices:
            global_idx = valid_indices[idx]
            atoms = structures[global_idx]

            try:
                gamma_arr = sampler.calculator.get_property("gamma", atoms)
                max_gamma_idx = int(np.argmax(gamma_arr)) if np.ndim(gamma_arr) > 0 else 0
                results.append((atoms, max_gamma_idx))
            except Exception as e:
                logger.warning(f"Failed to find max gamma for selected structure: {e}")
                results.append((atoms, 0))

        return results
