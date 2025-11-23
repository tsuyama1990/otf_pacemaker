"""Interfaces for the ACE Active Carver application.

This module defines the abstract base classes (protocols) for the core components
of the active learning loop, enabling dependency inversion and easier testing.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
from ase import Atoms
from src.core.enums import SimulationState


class MDEngine(ABC):
    """Interface for Molecular Dynamics engines."""

    @abstractmethod
    def run(
        self,
        potential_path: str,
        steps: int,
        gamma_threshold: float,
        input_structure: str,
        is_restart: bool = False,
    ) -> SimulationState:
        """Run the MD simulation."""
        pass


class Sampler(ABC):
    """Interface for uncertainty sampling strategies."""

    @abstractmethod
    def sample(self, **kwargs) -> List[Tuple[Atoms, int]]:
        """Select atoms/structures for training based on uncertainty.

        Args:
            **kwargs: Flexible arguments depending on the implementation.
                      e.g., atoms (Atoms object), dump_file (str), etc.

        Returns:
            List[Tuple[Atoms, int]]: A list of tuples, where each tuple contains
                                     the selected structure and the index of the
                                     atom with the highest uncertainty within it.
        """
        pass


class StructureGenerator(ABC):
    """Interface for generating training structures."""

    @abstractmethod
    def generate_cell(self, large_atoms: Atoms, center_id: int, potential_path: str) -> Atoms:
        """Generate a structure (e.g., small cell) around a center atom."""
        pass


class Labeler(ABC):
    """Interface for labeling structures (computing targets)."""

    @abstractmethod
    def label(self, structure: Atoms) -> Optional[Atoms]:
        """Compute target properties (e.g., forces, energy) for the structure."""
        pass


class Trainer(ABC):
    """Interface for training machine learning potentials."""

    @abstractmethod
    def prepare_dataset(self, structures: List[Atoms]) -> str:
        """Prepare the dataset for training."""
        pass

    @abstractmethod
    def update_active_set(self, dataset_path: str, potential_yaml_path: str) -> str:
        """Update the active set using the dataset and potential definition.

        Args:
            dataset_path: Path to the full training dataset.
            potential_yaml_path: Path to the potential basis set definition.

        Returns:
            str: Path to the updated active set file (.asi).
        """
        pass

    @abstractmethod
    def train(self, dataset_path: str, initial_potential: str, **kwargs) -> str:
        """Train the potential.

        Args:
            dataset_path: Path to the training dataset.
            initial_potential: Path to the initial potential (optional/nullable depending on implementation).
            **kwargs: Additional arguments like potential_yaml_path, asi_path, etc.

        Returns:
            str: Path to the trained potential file.
        """
        pass
