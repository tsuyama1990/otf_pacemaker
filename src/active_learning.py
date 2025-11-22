"""Active Learning module for cluster extraction.

This module provides functionality to extract atomic clusters from simulation
snapshots based on uncertainty criteria.
"""

import numpy as np
from ase import Atoms


class ClusterCarver:
    """Responsible for carving out clusters from atomic structures."""

    def extract_cluster(self, atoms: Atoms, center_id: int) -> tuple[Atoms, np.ndarray]:
        """Extract a cluster of atoms centered around a specific atom.

        Args:
            atoms: The full atomic structure (ASE Atoms object).
            center_id: The index of the center atom for the cluster.

        Returns:
            tuple[Atoms, np.ndarray]:
                - A new ASE Atoms object representing the extracted cluster.
                - A numpy array of force weights for each atom in the cluster
                  (Core=1.0, Buffer=0.0).
        """
        pass
