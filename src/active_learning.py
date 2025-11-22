"""Active Learning module for small-cell generation.

This module provides functionality to generate small periodic cells around
uncertain local environments for training.
"""

import logging
import numpy as np
from ase import Atoms
# ASE 3.23+ moved LAMMPS calculator or I might be using an older version logic.
# Let's check availability.
# In newer ASE, 'LAMMPS' is in ase.calculators.lammpsrun or ase.calculators.lammps.
# If ase.calculators.lammps doesn't have LAMMPS class, it might be in lammpsrun.
try:
    from ase.calculators.lammpsrun import LAMMPS
except ImportError:
    try:
        from ase.calculators.lammps import LAMMPS
    except ImportError:
        # Fallback or error, but let's try lammpslib if strictly needed, but we want file-based.
        raise ImportError("Could not import LAMMPS calculator from ase.calculators.lammpsrun or ase.calculators.lammps")

from ase.constraints import ExpCellFilter, FixAtoms
from ase.optimize import FIRE

logger = logging.getLogger(__name__)


class SmallCellGenerator:
    """Responsible for generating and relaxing small periodic cells."""

    def __init__(
        self,
        box_size: float,
        r_core: float,
        lammps_cmd: str = "lmp_serial",
        stoichiometry_tolerance: float = 0.1,
        elements: list[str] | None = None,
    ):
        """Initialize the SmallCellGenerator.

        Args:
            box_size: Size of the cubic small cell (Angstroms).
            r_core: Radius for the core region where atoms are fixed during relaxation.
            lammps_cmd: Command to run LAMMPS.
            stoichiometry_tolerance: Tolerance for stoichiometry warning.
            elements: List of element symbols (needed for potential mapping if generic types used).
        """
        self.box_size = box_size
        self.r_core = r_core
        self.lammps_cmd = lammps_cmd
        self.stoichiometry_tolerance = stoichiometry_tolerance
        self.elements = elements if elements else []

    def generate_cell(
        self, atoms: Atoms, center_id: int, current_potential: str
    ) -> Atoms:
        """Generate and relax a small periodic cell around a center atom.

        Args:
            atoms: The full atomic structure (ASE Atoms object).
            center_id: The index of the center atom.
            current_potential: Path to the current ACE potential file (.yace).

        Returns:
            Atoms: The relaxed small periodic cell.
        """
        # 1. Extraction & PBC Setup
        # We want to extract atoms within box_size from the center_id.
        # First, calculate vectors from center_id to all other atoms, respecting MIC of the original system.
        # We assume 'atoms' has a cell and pbc.
        if atoms.pbc.any() and atoms.cell.volume > 0:
            # Use get_distances with vector=True and mic=True
            # This returns vectors pointing FROM center_id TO other atoms.
            vectors = atoms.get_distances(
                center_id, range(len(atoms)), mic=True, vector=True
            )
        else:
            # Non-periodic or no cell, simple difference
            vectors = atoms.positions - atoms.positions[center_id]

        # Filter atoms that fall within the new box centered at origin
        # We want a cube of side box_size. So range is [-box_size/2, box_size/2].
        half_box = self.box_size / 2.0
        mask = (np.abs(vectors) <= half_box).all(axis=1)

        # Create the new Atoms object
        subset_atoms = atoms[mask].copy()
        subset_atoms.positions = vectors[mask]  # Positions relative to center (at 0,0,0)

        # Shift atoms so the center atom is at the center of the new box
        subset_atoms.positions += half_box

        # Set the new cell and PBC
        subset_atoms.set_cell([self.box_size, self.box_size, self.box_size])
        subset_atoms.set_pbc(True)

        # Wrap atoms to be inside the box [0, box_size]
        subset_atoms.wrap()

        # 2. Stoichiometry Check
        self._check_stoichiometry(atoms, subset_atoms)

        # 3. MLIP Constrained Relaxation
        relaxed_atoms = self._relax_cell(subset_atoms, current_potential)

        return relaxed_atoms

    def _check_stoichiometry(self, original: Atoms, subset: Atoms):
        """Check if the stoichiometry of the subset matches the original within tolerance."""
        if not len(original) or not len(subset):
            return

        orig_symbols = original.get_chemical_symbols()
        sub_symbols = subset.get_chemical_symbols()

        unique_elements = sorted(list(set(orig_symbols)))

        for el in unique_elements:
            orig_frac = orig_symbols.count(el) / len(original)
            sub_frac = sub_symbols.count(el) / len(subset)

            if abs(orig_frac - sub_frac) > self.stoichiometry_tolerance:
                logger.warning(
                    f"Stoichiometry mismatch for {el}: Original {orig_frac:.2f}, Subset {sub_frac:.2f}"
                )

    def _relax_cell(self, atoms: Atoms, potential_path: str) -> Atoms:
        """Relax the cell using the given potential with constraints.

        Args:
            atoms: The atoms to relax.
            potential_path: Path to the .yace potential.

        Returns:
            Atoms: The relaxed structure.
        """
        # Setup LAMMPS Calculator
        # pair_style pace ...
        # We need to handle species mapping.
        # If self.elements is set, we use it.

        element_map = " ".join(self.elements) if self.elements else " ".join(sorted(list(set(atoms.get_chemical_symbols()))))

        # LAMMPS input parameters
        parameters = {
            "pair_style": "pace",
            "pair_coeff": [f"* * {potential_path} {element_map}"],
            "mass": ["* 1.0"], # Simplified mass
        }

        calc = LAMMPS(
            command=self.lammps_cmd,
            parameters=parameters,
            specorder=self.elements if self.elements else None,
            files=[potential_path], # Ensure potential file is copied if needed
            keep_tmp_files=False, # Clean up
        )

        atoms.calc = calc

        # Constraint: Fix atoms within r_core from the center of the box
        # Center is at [box_size/2, box_size/2, box_size/2]
        box_center = np.array([self.box_size / 2.0] * 3)

        # Calculate distances from box center
        # We compute relative positions manually to handle PBC
        rel_pos = atoms.positions - box_center

        # Apply MIC
        # Since cell is orthogonal cubic (box_size^3), we can just use modulo/round
        # or simply assume wrap() put them in [0, box_size], so closest image to center [L/2]
        # is just diff.
        # Example: atom at 0.1, center at 5.0. Dist 4.9.
        # Example: atom at 9.9, center at 5.0. Dist 4.9.
        # We need standard MIC.

        # ASE Atoms doesn't have simple MIC for arbitrary point without using get_distances hacks
        # or cell.
        # But we know the cell is orthogonal.
        L = self.box_size
        rel_pos = rel_pos - L * np.round(rel_pos / L)
        dists = np.linalg.norm(rel_pos, axis=1)

        fixed_indices = np.where(dists < self.r_core)[0]

        if len(fixed_indices) > 0:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
            logger.info(f"Fixed {len(fixed_indices)} atoms within core radius {self.r_core}")
        else:
            logger.warning("No atoms found within r_core to fix.")

        # Relaxation
        # Use ExpCellFilter to relax positions and cell
        try:
            ucf = ExpCellFilter(atoms)
            opt = FIRE(ucf, logfile=None) # Suppress log or redirect
            opt.run(fmax=0.05, steps=100) # Reasonable limit
        except Exception as e:
            logger.error(f"Relaxation failed: {e}")
            # Return unrelaxed atoms or partial result?
            # Ideally return atoms as is if failed, but warn.
            pass

        return atoms
