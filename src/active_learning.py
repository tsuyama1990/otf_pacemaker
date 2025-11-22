"""Active Learning module for cluster extraction.

This module provides functionality to extract atomic clusters from simulation
snapshots based on uncertainty criteria.
"""

import numpy as np
import pandas as pd
from ase import Atoms
from ase.neighborlist import neighbor_list


class ClusterCarver:
    """Responsible for carving out clusters from atomic structures."""

    def __init__(self, r_core: float, r_buffer: float):
        """Initialize the ClusterCarver.

        Args:
            r_core: Radius for the core region where forces are fully weighted.
            r_buffer: Radius for the buffer region (total cluster radius).
        """
        self.r_core = r_core
        self.r_buffer = r_buffer

    def extract_cluster(self, atoms: Atoms, center_id: int) -> Atoms:
        """Extract a cluster of atoms centered around a specific atom.

        The cluster contains all atoms within `r_buffer` from the center atom.
        A `forces_weight` array is calculated and stored in the returned Atoms object,
        assigning 1.0 to atoms within `r_core` and 0.0 to atoms in the buffer region.

        Args:
            atoms: The full atomic structure (ASE Atoms object).
            center_id: The index of the center atom for the cluster.

        Returns:
            Atoms: A new ASE Atoms object representing the extracted cluster.
                The object includes:
                - arrays['forces_weight']: Weight for each atom (1.0 or 0.0).
        """
        # Find neighbors within r_buffer
        # We use neighbor_list to get distances and indices
        # 'd': distances, 'j': indices of neighbors
        # We only need neighbors of the center_id.
        # However, neighbor_list typically works on the whole system or we filter.
        # A more efficient way for a single atom might be to just calculate distances
        # if the system is small, or use the neighbor list if it is large.
        # Given ASE, we can use atoms.get_distances() but that accounts for MIC only if cell is set.

        # Let's use a simple approach: calculate distances from center_id to all others.
        # This handles MIC if pbc is True.
        distances = atoms.get_distances(center_id, range(len(atoms)), mic=True)

        # Identify indices within r_buffer
        mask_buffer = distances <= self.r_buffer
        indices = np.where(mask_buffer)[0]

        # Extract the cluster
        cluster = atoms[indices]

        # Calculate weights for the extracted cluster
        # The distances array needs to be sliced to match the cluster atoms
        cluster_distances = distances[indices]

        weights = np.where(cluster_distances <= self.r_core, 1.0, 0.0)

        # Assign weights to the cluster
        cluster.arrays['forces_weight'] = weights

        return cluster


def convert_to_pacemaker_format(atoms_list: list[Atoms]) -> pd.DataFrame:
    """Convert a list of ASE Atoms objects to a Pacemaker-compatible DataFrame.

    The returned DataFrame has a single column 'ase_atoms' containing the
    Atoms objects.

    Args:
        atoms_list: List of ASE Atoms objects.

    Returns:
        pd.DataFrame: A DataFrame suitable for Pacemaker training.
    """
    return pd.DataFrame({"ase_atoms": atoms_list})
