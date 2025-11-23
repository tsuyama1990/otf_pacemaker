"""Filter module using MACE foundation models.

This module provides functionality to filter atomic structures based on
energy and force predictions from a pre-trained MACE model.
"""

import logging
from typing import List
import numpy as np
from ase import Atoms

try:
    from mace.calculators import mace_mp
except ImportError:
    mace_mp = None

logger = logging.getLogger(__name__)


class MACEFilter:
    """Filters structures using MACE foundation model predictions."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        energy_cutoff_per_atom: float = 0.0,
        force_cutoff: float = 100.0 # e.g. eV/A
    ):
        """Initialize the MACEFilter.

        Args:
            model_size: Size of the MACE model ("small", "medium", "large").
            device: Device to run the model on ("cpu", "cuda").
            energy_cutoff_per_atom: Threshold for energy per atom (deprecated logic usually, checking for physical validity).
                                    Actually, high positive energy usually means bad structure.
                                    However, raw energy depends on reference.
                                    Usually we look for forces to be reasonable.
            force_cutoff: Maximum allowed force component (or norm) on any atom.
                          Structures with forces higher than this are likely unphysical overlaps.
        """
        self.model_size = model_size
        self.device = device
        self.energy_cutoff_per_atom = energy_cutoff_per_atom
        self.force_cutoff = force_cutoff
        self.calc = None

    def _load_model(self):
        """Lazy load the MACE calculator."""
        if self.calc is None:
            if mace_mp is None:
                raise ImportError("MACE is required. Please install 'mace-torch'.")
            # Using default wrappers which handle download
            self.calc = mace_mp(model=self.model_size, device=self.device, default_dtype="float64")

    def filter(self, structures: List[Atoms]) -> List[Atoms]:
        """Filter out unphysical structures.

        Args:
            structures: List of candidate structures.

        Returns:
            List[Atoms]: Filtered list of valid structures.
        """
        self._load_model()
        valid_structures = []

        for atoms in structures:
            try:
                # Perform calculation
                atoms.calc = self.calc
                # We trigger calculation by getting forces
                forces = atoms.get_forces()
                energy = atoms.get_potential_energy()

                # Check forces
                # If max force is too high, it's likely an overlap
                max_force = np.max(np.linalg.norm(forces, axis=1))

                if max_force > self.force_cutoff:
                    logger.debug(f"Structure rejected: Max force {max_force:.2f} > {self.force_cutoff}")
                    continue

                # Check energy
                # Very high positive energy per atom often indicates issues,
                # but absolute energy depends on MACE's reference.
                # We will trust force mainly, but can check if energy is NaN.
                if np.isnan(energy):
                    continue

                # If needed we can add energy/atom threshold if we have a reference.
                # For now relying on force and lack of errors.

                # Store MACE results if useful later, or clear calc to save memory/disk space
                # Clearing calc to avoid pickling issues or dependency on MACE later
                atoms.calc = None
                # We might want to store MACE energy/forces in info/arrays?
                # Requirement says "filtering", doesn't strictly say we need to keep the labels.
                # We will label with DFT later.

                atoms.info['mace_energy'] = energy
                atoms.arrays['mace_forces'] = forces

                valid_structures.append(atoms)

            except Exception as e:
                logger.warning(f"MACE calculation failed for a structure: {e}")
                continue

        logger.info(f"Filtered {len(structures)} -> {len(valid_structures)} structures.")
        return valid_structures
