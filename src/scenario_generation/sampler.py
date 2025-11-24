"""Sampling module using DIRECT strategy (ACE descriptors + BIRCH).

This module selects a diverse set of structures from a large pool
using clustering on ACE descriptors.
"""

import logging
import tempfile
from pathlib import Path
from typing import List
import numpy as np
import yaml
from ase import Atoms

from src.sampling.strategies.ace_sampler import ACESampler

try:
    from sklearn.cluster import Birch
except ImportError:
    Birch = None

logger = logging.getLogger(__name__)


class DirectSampler:
    """Selects diverse structures using ACE descriptors and BIRCH clustering."""

    def __init__(self, n_clusters: int = 200):
        """Initialize the DirectSampler.

        Args:
            n_clusters: The target number of structures to select.
        """
        self.n_clusters = n_clusters

    def _generate_potential_yaml(self, elements: List[str], output_path: Path):
        """Generates a minimal potential.yaml for descriptor calculation.

        This assumes a default basis set configuration.
        """
        # Minimal configuration for ACE descriptors
        config = {
            "cutoff": 6.0,  # Reasonable default
            "elements": elements,
            "bonds": {
                "N": 3,  # Max body order (N=3 -> 4-body)
                "max_deg": 10,
                "r0": 1.5,
                "rad_base": "Chebyshev",
                "rad_parameters": [10.0],
            },
            "embeddings": {
                 el: {
                     "npot": "FinnisSinclair",
                     "fs_parameters": [1, 1, 1, 0.5],
                     "ndensity": 1,
                 } for el in elements
            }
        }

        with open(output_path, "w") as f:
            yaml.dump(config, f)

    def sample(self, structures: List[Atoms]) -> List[Atoms]:
        """Selects a diverse subset of structures using ACE descriptors.

        Args:
            structures: The pool of candidate structures.

        Returns:
            List[Atoms]: The selected subset of structures.
        """
        if Birch is None:
            raise ImportError("scikit-learn is required. Please install 'scikit-learn'.")

        if not structures:
            return []

        if len(structures) <= self.n_clusters:
            logger.info(f"Number of structures ({len(structures)}) <= requested clusters ({self.n_clusters}). Returning all.")
            return structures

        # 1. Prepare for ACE descriptor calculation
        all_elements = sorted(list(set([
            s for atoms in structures for s in atoms.get_chemical_symbols()
        ])))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pot_path = Path(temp_dir) / "potential.yaml"
            self._generate_potential_yaml(all_elements, temp_pot_path)

            try:
                ace_sampler = ACESampler(str(temp_pot_path))
            except ImportError as e:
                logger.error(f"ACESampler initialization failed: {e}")
                raise e

            logger.info("Computing ACE descriptors for sampling...")
            descriptors_list = []

            for atoms in structures:
                try:
                    desc = ace_sampler.compute_descriptors(atoms)

                    if desc.ndim == 2:
                        # (n_atoms, n_features) -> (n_features,)
                        desc = np.mean(desc, axis=0)
                    elif desc.ndim == 1:
                        pass

                    descriptors_list.append(desc)
                except Exception as e:
                    logger.warning(f"Descriptor calculation failed for structure: {e}. Skipping.")
                    descriptors_list.append(None)

        # Filter out failed calculations
        valid_indices = [i for i, d in enumerate(descriptors_list) if d is not None]
        X = np.array([descriptors_list[i] for i in valid_indices])
        valid_structures = [structures[i] for i in valid_indices]

        if len(valid_structures) < self.n_clusters:
            logger.warning("Valid structures < n_clusters after descriptor calc. Returning valid ones.")
            return valid_structures

        logger.info(f"Clustering {len(valid_structures)} structures using ACE descriptors...")

        # BIRCH clustering
        brc = Birch(n_clusters=self.n_clusters)
        labels = brc.fit_predict(X)

        selected_structures = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            indices = np.where(labels == label)[0]

            if len(indices) == 1:
                selected_structures.append(valid_structures[indices[0]])
            else:
                cluster_X = X[indices]
                centroid = np.mean(cluster_X, axis=0)
                dists = np.linalg.norm(cluster_X - centroid, axis=1)
                closest_idx_in_cluster = np.argmin(dists)
                global_idx = indices[closest_idx_in_cluster]
                selected_structures.append(valid_structures[global_idx])

        logger.info(f"Selected {len(selected_structures)} structures.")
        return selected_structures
