"""Sampling module using DIRECT strategy (ACE descriptors + BIRCH).

This module selects a diverse set of structures from a large pool
using clustering on ACE descriptors.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
import numpy as np
import yaml
from ase import Atoms
from ase.io import write

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
        # We don't need accurate coefficients, just the basis definition.
        config = {
            "cutoff": 6.0,  # Reasonable default
            "elements": elements,
            "bonds": {
                "N": 3,  # Max body order (N=3 -> 4-body)
                "max_deg": 10,
                "r0": 1.5,
                "rad_base": "Chebyshev",
                "rad_parameters": [10.0], # Only cutoff needed for Cheb
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

    def _compute_descriptors(self, structures: List[Atoms]) -> np.ndarray:
        """Computes structural descriptors for the given structures.

        Uses simple structural features (composition, volume, density, radial distribution)
        since pace_collect CLI tool may not be available.
        """
        if not structures:
            return np.array([])

        logger.info("Computing structural descriptors (composition + geometry features)...")
        
        descriptors = []
        
        # Get all unique elements across all structures
        all_elements = sorted(list(set([
            s for atoms in structures for s in atoms.get_chemical_symbols()
        ])))
        
        for atoms in structures:
            features = []
            
            # 1. Basic structural features
            n_atoms = len(atoms)
            volume = atoms.get_volume() if atoms.cell.volume > 0 else 0
            density = n_atoms / volume if volume > 0 else 0
            
            features.extend([n_atoms, volume, density])
            
            # 2. Composition vector (normalized counts for each element)
            symbols = atoms.get_chemical_symbols()
            composition = [symbols.count(el) / n_atoms for el in all_elements]
            features.extend(composition)
            
            # 3. Geometric features
            try:
                positions = atoms.get_positions()
                if len(positions) > 1:
                    # Center of mass
                    com = np.mean(positions, axis=0)
                    # Radius of gyration
                    rg = np.sqrt(np.mean(np.sum((positions - com)**2, axis=1)))
                    # Average nearest neighbor distance
                    from scipy.spatial import distance_matrix
                    dist_mat = distance_matrix(positions, positions)
                    np.fill_diagonal(dist_mat, np.inf)
                    avg_nn_dist = np.mean(np.min(dist_mat, axis=1))
                else:
                    rg = 0.0
                    avg_nn_dist = 0.0
                    
                features.extend([rg, avg_nn_dist])
            except Exception as e:
                logger.debug(f"Could not compute geometric features: {e}")
                features.extend([0.0, 0.0])
            
            # 4. Cell shape features (if periodic)
            if atoms.cell.volume > 0:
                cell_lengths = atoms.cell.lengths()
                cell_angles = atoms.cell.angles()
                features.extend(list(cell_lengths))
                features.extend(list(cell_angles))
            else:
                features.extend([0.0] * 6)
            
            descriptors.append(features)
        
        descriptors_array = np.array(descriptors)
        logger.info(f"Computed {descriptors_array.shape[1]} features for {len(structures)} structures")
        
        return descriptors_array

    def sample(self, structures: List[Atoms]) -> List[Atoms]:
        """Selects a diverse subset of structures.

        Args:
            structures: The pool of candidate structures.

        Returns:
            List[Atoms]: The selected subset of structures.
        """
        if Birch is None:
            raise ImportError("scikit-learn is required. Please install 'scikit-learn'.")

        if len(structures) <= self.n_clusters:
            logger.info(f"Number of structures ({len(structures)}) <= requested clusters ({self.n_clusters}). Returning all.")
            return structures

        logger.info("Computing descriptors for sampling...")
        X = self._compute_descriptors(structures)

        if X.shape[0] != len(structures):
             logger.error("Descriptor count mismatch. Returning random selection.")
             # Fallback to random
             import random
             return random.sample(structures, self.n_clusters)

        logger.info(f"Clustering {len(structures)} structures into {self.n_clusters} clusters...")

        # BIRCH clustering
        # branching_factor and threshold might need tuning, but defaults often work or we specify n_clusters directly
        brc = Birch(n_clusters=self.n_clusters)
        labels = brc.fit_predict(X)

        selected_structures = []

        # Stratified sampling: pick one from each cluster
        # To be 'representative', we could pick the one closest to cluster centroid.
        # BIRCH stores subclusters, but let's just pick the one closest to the mean of the cluster members in X.

        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = np.where(labels == label)[0]

            if len(indices) == 1:
                selected_structures.append(structures[indices[0]])
            else:
                # Calculate centroid of this cluster
                cluster_X = X[indices]
                centroid = np.mean(cluster_X, axis=0)

                # Find index of sample closest to centroid
                dists = np.linalg.norm(cluster_X - centroid, axis=1)
                closest_idx_in_cluster = np.argmin(dists)
                global_idx = indices[closest_idx_in_cluster]

                selected_structures.append(structures[global_idx])

        logger.info(f"Selected {len(selected_structures)} structures.")
        return selected_structures
