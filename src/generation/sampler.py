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
        """Computes ACE descriptors for the given structures.

        Uses 'pace_collect' or similar tool via subprocess since we need
        averaged descriptors (structure vectors) for clustering.
        """
        if not structures:
            return np.array([])

        # Identify elements
        elements = sorted(list(set([s for atoms in structures for s in atoms.get_chemical_symbols()])))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Write structures to file
            data_path = tmp_path / "data.pckl.gzip"
            # We need a format pacemaker handles. pandas pickle with 'ase_atoms' column is standard here.
            import pandas as pd
            df = pd.DataFrame({"ase_atoms": structures})
            df.to_pickle(data_path, compression="gzip")

            # 2. Generate potential.yaml
            pot_yaml_path = tmp_path / "potential.yaml"
            self._generate_potential_yaml(elements, pot_yaml_path)

            # 3. Run pace_collect to compute B matrix (descriptors)
            # usage: pace_collect <potential_file> <dataset_file> --output_file <output> --compute_descriptors
            # But pace_collect usually computes B matrix for linear fitting.
            # We want the structure descriptors.
            # Actually `pace_collect` produces `collected_data.pckl` which contains 'design_matrix'.
            # The design matrix rows correspond to structures (if global) or atoms?
            # ACE is usually atomic. We want structure descriptors.
            # Standard approach: Sum/Mean of atomic descriptors.
            # pace_collect returns design matrix which is summed over atoms for total energy fitting.
            # So one row per structure. That is exactly what we need for clustering structures.

            output_pckl = tmp_path / "descriptors.pckl"

            cmd = [
                "pace_collect",
                str(pot_yaml_path),
                str(data_path),
                "--output_file", str(output_pckl)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"pace_collect failed: {e.stderr.decode()}")
                # Fallback or raise?
                # If pace_collect fails, we can't compute descriptors.
                raise RuntimeError("Failed to compute ACE descriptors.")

            # 4. Load descriptors
            # The output is a pickled dataframe/dict usually containing the design matrix
            # "design_matrix" is usually (N_structures, N_features)
            data = pd.read_pickle(output_pckl)
            if "design_matrix" in data:
                descriptors = data["design_matrix"]
                # Ensure it's numpy array
                if hasattr(descriptors, "toarray"): # sparse
                    descriptors = descriptors.toarray()
                return np.array(descriptors)
            else:
                 # Check structure
                 raise ValueError(f"Unexpected output format from pace_collect: {data.keys()}")

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
