"""Systematic Deformation Generator for EOS and Elastic Properties."""

import numpy as np
from typing import List, Optional
from ase import Atoms
from src.core.interfaces import StructureGenerator
from src.core.config import LJParams

class SystematicDeformationGenerator:
    """Generates distorted structures for Equation of State and Elastic constants learning."""

    def __init__(self, current_structure: Atoms, lj_params: LJParams):
        """Initialize the generator.

        Args:
            current_structure: The reference structure to deform.
            lj_params: LJ parameters (passed for consistency if needed, though mostly geometric).
        """
        self.ref_structure = current_structure.copy()
        self.lj_params = lj_params

    def generate_hydrostatic(self, scales: List[float] = None) -> List[Atoms]:
        """Generate hydrostatically strained structures.

        Args:
            scales: List of volume scaling factors (e.g., 0.94 for 6% compression).
                    Default: [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]

        Returns:
            List of deformed Atoms objects.
        """
        if scales is None:
            scales = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]

        deformed_structures = []
        for scale in scales:
            atoms = self.ref_structure.copy()
            # scale_atoms=True scales positions as well as cell
            # But set_cell scaling is usually linear dimension scaling or volume?
            # ASE set_cell 'scale_atoms' scales positions.
            # If we pass a new cell, it sets it.
            # To scale volume by X, we scale lengths by X^(1/3).

            # However, prompt said "Takes a list of scale factors... Uses atoms.set_cell(..., scale_atoms=True)".
            # Usually "scale factor" implies linear scaling in crystallography, but for EOS usually Volume scaling.
            # "Verify volume of result is 0.9^3..." implies linear scaling factor is passed.
            # If prompt says "0.94", it likely means linear scale.

            current_cell = atoms.get_cell()
            new_cell = current_cell * scale
            atoms.set_cell(new_cell, scale_atoms=True)

            # Tag it
            atoms.info['deformation'] = f"hydro_{scale:.2f}"
            deformed_structures.append(atoms)

        return deformed_structures

    def generate_shear(self, max_shear: float = 0.05, n_steps: int = 5) -> List[Atoms]:
        """Generate sheared structures.

        Applies shear strains to the cell.
        We apply simple monoclinic shears (e.g. xy, xz, yz).

        Args:
            max_shear: Maximum shear strain (gamma).
            n_steps: Number of steps from -max to +max (excluding 0 if redundant).

        Returns:
            List of sheared Atoms objects.
        """
        # Linear spacing
        shears = np.linspace(-max_shear, max_shear, n_steps)
        # Remove near-zero to avoid duplicating reference
        shears = [s for s in shears if abs(s) > 1e-6]

        deformed_structures = []
        cell = self.ref_structure.get_cell()

        # Strain matrices for shears
        # e.g. xy shear:
        # | 1  g  0 |
        # | 0  1  0 |
        # | 0  0  1 |

        for g in shears:
            # xy shear
            strain_xy = np.eye(3)
            strain_xy[0, 1] = g

            # xz shear
            strain_xz = np.eye(3)
            strain_xz[0, 2] = g

            # yz shear
            strain_yz = np.eye(3)
            strain_yz[1, 2] = g

            for tag, strain in [("xy", strain_xy), ("xz", strain_xz), ("yz", strain_yz)]:
                atoms = self.ref_structure.copy()
                # New cell = Cell @ Strain (ASE conventions vary, but usually row vectors)
                # cell is 3x3 array of row vectors.
                # new_cell = cell @ strain
                new_cell = np.dot(cell, strain)

                # We need to map positions to new cell
                # Fractionals stay same
                scaled_pos = atoms.get_scaled_positions()
                atoms.set_cell(new_cell)
                atoms.set_scaled_positions(scaled_pos)

                atoms.info['deformation'] = f"shear_{tag}_{g:.3f}"
                deformed_structures.append(atoms)

        return deformed_structures

    def generate_all(self) -> List[Atoms]:
        """Generate both hydrostatic and shear deformations."""
        return self.generate_hydrostatic() + self.generate_shear()
