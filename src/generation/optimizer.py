"""Module for structure optimization using foundation models (MACE)."""

import logging
from typing import List, Optional

from ase import Atoms
from ase.optimize import LBFGS

try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    mace_mp = None

logger = logging.getLogger(__name__)


class FoundationOptimizer:
    """Optimizer using MACE foundation model to relax structures."""

    def __init__(
        self,
        model: str = "medium",
        device: str = "cuda",
        fmax: float = 0.1,
        steps: int = 50,
    ):
        """Initialize the FoundationOptimizer.

        Args:
            model: Size of the MACE model ("small", "medium", "large").
            device: Device to run the model on ("cpu", "cuda").
            fmax: Force convergence criterion in eV/A.
            steps: Maximum number of optimization steps.

        Raises:
            ImportError: If mace-torch is not installed.
        """
        if not MACE_AVAILABLE:
            raise ImportError(
                "mace-torch is not installed. Please install it to use FoundationOptimizer."
            )

        try:
            self.calc = mace_mp(model=model, device=device, default_dtype="float32")
        except Exception as e:
            # Fallback or re-raise if model loading fails
            logger.error(f"Failed to initialize MACE calculator: {e}")
            raise

        self.fmax = fmax
        self.steps = steps

    def relax(self, structures: List[Atoms]) -> List[Atoms]:
        """Relax a list of structures using MACE.

        Args:
            structures: List of Atoms objects to relax.

        Returns:
            List[Atoms]: List of successfully relaxed structures.
        """
        relaxed_structures = []
        for atoms in structures:
            try:
                atoms_copy = atoms.copy()
                atoms_copy.calc = self.calc
                # Suppress log output
                opt = LBFGS(atoms_copy, logfile=None)
                opt.run(fmax=self.fmax, steps=self.steps)

                # Sanity check
                if self._is_physically_valid(atoms_copy):
                    relaxed_structures.append(atoms_copy)
            except Exception as e:
                logger.warning(f"Pre-optimization failed for structure: {e}")
                continue
        return relaxed_structures

    def _is_physically_valid(self, atoms: Atoms) -> bool:
        """Check if the structure is physically valid.

        Currently a placeholder for energy divergence or atom collapse checks.

        Args:
            atoms: Atoms object to check.

        Returns:
            bool: True if valid, False otherwise.
        """
        # TODO: Implement actual physical validation (e.g., min distance check)
        return True
