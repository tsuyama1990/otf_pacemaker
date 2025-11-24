"""ACESampler strategy using PyACE native bindings."""

import logging
import numpy as np
from typing import List, Optional
from ase import Atoms

try:
    import pyace
except ImportError:
    pyace = None

logger = logging.getLogger(__name__)

class ACESampler:
    """Computes ACE descriptors and extrapolation grades using PyACE."""

    def __init__(self, potential_path: str):
        """Initialize ACESampler.

        Args:
            potential_path: Path to the potential YAML or YACE file.
        """
        if pyace is None:
            raise ImportError("pyace module is required for ACESampler.")

        self.potential_path = potential_path
        self._calculator = None

    @property
    def calculator(self):
        """Lazy initialization of PyACECalculator."""
        if self._calculator is None:
            logger.info(f"Initializing PyACECalculator from {self.potential_path}")
            # Assuming PyACECalculator takes the potential path as an argument
            self._calculator = pyace.PyACECalculator(self.potential_path)
        return self._calculator

    def compute_descriptors(self, atoms: Atoms) -> np.ndarray:
        """Compute B-basis vector (descriptors) for the given structure.

        Args:
            atoms: ASE Atoms object.

        Returns:
            np.ndarray: Array of descriptors.
        """
        try:
            descriptors = self.calculator.get_property("B", atoms)
            return np.array(descriptors)
        except Exception as e:
            logger.error(f"Failed to compute descriptors: {e}")
            raise e

    def compute_gamma(self, atoms: Atoms) -> float:
        """Compute extrapolation grade (gamma) for the given structure.

        Args:
            atoms: ASE Atoms object.

        Returns:
            float: The maximum gamma value in the structure.
        """
        try:
            gamma_values = self.calculator.get_property("gamma", atoms)
            if np.ndim(gamma_values) > 0:
                return float(np.max(gamma_values))
            return float(gamma_values)
        except Exception as e:
            logger.error(f"Failed to compute gamma: {e}")
            raise e
