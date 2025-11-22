"""Training module for Pacemaker potentials.

This module handles the preparation of datasets and training of ACE potentials
using the Pacemaker library.
"""

from pathlib import Path
from ase import Atoms


class PacemakerTrainer:
    """Manages the training process for ACE potentials."""

    def prepare_dataset(self, atoms_list: list[Atoms]) -> str:
        """Convert a list of labeled Atoms objects into a training dataset.

        The method should serialize the data into a format suitable for Pacemaker
        (e.g., a pandas DataFrame saved as a pickle file).

        Args:
            atoms_list: A list of labeled ASE Atoms objects (containing delta energy/forces).

        Returns:
            str: The file path to the created dataset (e.g., 'data.pckl.gzip').
        """
        pass

    def train(self, dataset_path: str, initial_potential: str) -> str:
        """Train the potential using Pacemaker.

        This method executes the training process, potentially using `subprocess`
        to call Pacemaker CLI tools.

        Args:
            dataset_path: Path to the training dataset file.
            initial_potential: Path to the initial potential file (or configuration) to start from.

        Returns:
            str: The path to the newly trained potential file.
        """
        pass
