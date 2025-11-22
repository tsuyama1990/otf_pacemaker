"""Interfaces for the ACE Active Carver application.

This module defines the abstract base classes (protocols) for the core components
of the active learning loop, enabling dependency inversion and easier testing.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from ase import Atoms
from src.enums import SimulationState


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
    def sample(self, atoms: Atoms, n_clusters: int) -> List[int]:
        """Select atom indices for training based on uncertainty."""
        pass


class StructureGenerator(ABC):
    """Interface for generating training structures."""

    @abstractmethod
    def generate(self, atoms: Atoms, center_id: int, potential_path: str) -> Atoms:
        """Generate a structure (e.g., small cell) around a center atom."""
        pass


class Labeler(ABC):
    """Interface for labeling structures (computing targets)."""

    @abstractmethod
    def label(self, structure: Atoms) -> Atoms:
        """Compute target properties (e.g., forces, energy) for the structure."""
        pass


class Trainer(ABC):
    """Interface for training machine learning potentials."""

    @abstractmethod
    def prepare_dataset(self, structures: List[Atoms]) -> str:
        """Prepare the dataset for training."""
        pass

    @abstractmethod
    def train(self, dataset_path: str, initial_potential: str) -> str:
        """Train the potential."""
        pass
