"""Candidate generation module using PyXtal.

This module provides functionality to generate random atomic structures
using the PyXtal library.
"""

import logging
import random
from typing import List
from ase import Atoms

try:
    from pyxtal import pyxtal
except ImportError:
    pyxtal = None

logger = logging.getLogger(__name__)


class RandomStructureGenerator:
    """Generates random atomic structures using PyXtal."""

    def __init__(self, elements: List[str], max_atoms: int = 8):
        """Initialize the RandomStructureGenerator.

        Args:
            elements: List of chemical elements to use (e.g., ["Al", "Cu"]).
            max_atoms: Maximum number of atoms in the generated unit cell.
        """
        self.elements = elements
        self.max_atoms = max_atoms

    def generate(self, n_structures: int) -> List[Atoms]:
        """Generate a list of random structures.

        Args:
            n_structures: The number of structures to generate.

        Returns:
            List[Atoms]: A list of ASE Atoms objects representing the random structures.

        Raises:
            ImportError: If PyXtal is not installed.
        """
        if pyxtal is None:
            raise ImportError("PyXtal is required for structure generation. Please install 'pyxtal'.")

        structures = []
        count = 0

        # Assuming 3D crystals
        dim = 3

        while count < n_structures:
            try:
                # Initialize PyXtal crystal
                struc = pyxtal()

                # Randomly select space group (1-230 for 3D)
                # PyXtal handles random generation internally if we pass just group number
                # However, pyxtal.from_random helps too.
                # Let's use pyxtal.from_random logic indirectly or directly.

                # We need to define composition.
                # Since we want random, we can pick random number of atoms
                # respecting max_atoms and elements.

                # Simple strategy: Randomly pick total atoms between 2 and max_atoms
                # Then randomly distribute them among elements.
                n_atoms = random.randint(2, self.max_atoms)

                # Distribute n_atoms among self.elements
                composition = []
                remaining = n_atoms
                for i in range(len(self.elements) - 1):
                    if remaining > 0:
                        num = random.randint(1, remaining)
                        composition.append(num)
                        remaining -= num
                    else:
                        composition.append(0)
                composition.append(remaining)

                # Filter out 0s if PyXtal doesn't like them (it usually takes a list of species count)
                # But we need corresponding species list.

                species = self.elements
                numIons = composition

                # Clean up species with 0 atoms
                active_species = [s for s, n in zip(species, numIons) if n > 0]
                active_counts = [n for n in numIons if n > 0]

                if not active_species:
                    continue

                # Random space group
                sg = random.randint(1, 230)

                struc.from_random(dim, sg, active_species, active_counts)

                if struc.valid:
                    ase_atoms = struc.to_ase()
                    structures.append(ase_atoms)
                    count += 1

            except Exception as e:
                logger.debug(f"Failed to generate structure: {e}")
                continue

        return structures
