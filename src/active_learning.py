"""Active Learning components for structure generation and sampling.

This module provides functionality to generate small periodic cells around
uncertain local environments and to sample those environments.
"""

import logging
import numpy as np
from typing import List, Optional

from ase import Atoms
from ase.constraints import ExpCellFilter, FixAtoms
from ase.optimize import FIRE

# Try importing LAMMPS calculator
try:
    from ase.calculators.lammpsrun import LAMMPS
except ImportError:
    try:
        from ase.calculators.lammps import LAMMPS
    except ImportError:
        raise ImportError("Could not import LAMMPS calculator from ase.calculators.lammpsrun or ase.calculators.lammps")

from src.interfaces import StructureGenerator, Sampler

logger = logging.getLogger(__name__)


class MaxGammaSampler(Sampler):
    """Selects atoms with the highest uncertainty (gamma) values."""

    def sample(self, atoms: Atoms, n_clusters: int) -> List[int]:
        """Select atom indices with the maximum gamma values.

        Args:
            atoms: The Atoms object containing 'f_f_gamma' or similar array.
            n_clusters: The number of atoms to select.

        Returns:
            List[int]: Indices of the selected atoms.
        """
        # Try to find the gamma array
        gamma_key = None
        for key in atoms.arrays.keys():
            if "gamma" in key:
                gamma_key = key
                break

        if not gamma_key:
            logger.warning("Gamma values not found in atoms.arrays. Falling back to random selection.")
            rng = np.random.default_rng()
            return list(rng.choice(len(atoms), size=min(n_clusters, len(atoms)), replace=False))

        gammas = atoms.get_array(gamma_key)
        if gammas.ndim > 1:
            gammas = gammas.flatten()

        # Get indices of top n_clusters gammas
        # np.argsort returns ascending, so we take the last n
        sorted_indices = np.argsort(gammas)
        top_indices = sorted_indices[-n_clusters:]

        # Reverse to have highest first
        top_indices = top_indices[::-1]

        logger.info(f"Selected {len(top_indices)} atoms with max gamma {gammas[top_indices[0]]:.4f}")

        return [int(idx) for idx in top_indices]


class SmallCellGenerator(StructureGenerator):
    """Responsible for generating and relaxing small periodic cells."""

    def __init__(
        self,
        box_size: float,
        r_core: float,
        lammps_cmd: str = "lmp_serial",
        stoichiometry_tolerance: float = 0.1,
        elements: Optional[List[str]] = None,
    ):
        """Initialize the SmallCellGenerator.

        Args:
            box_size: Size of the cubic small cell (Angstroms).
            r_core: Radius for the core region where atoms are fixed during relaxation.
            lammps_cmd: Command to run LAMMPS.
            stoichiometry_tolerance: Tolerance for stoichiometry warning.
            elements: List of element symbols.
        """
        self.box_size = box_size
        self.r_core = r_core
        self.lammps_cmd = lammps_cmd
        self.stoichiometry_tolerance = stoichiometry_tolerance
        self.elements = elements if elements else []

    def generate(self, atoms: Atoms, center_id: int, potential_path: str) -> Atoms:
        """Generate and relax a small periodic cell around a center atom.

        Args:
            atoms: The full atomic structure.
            center_id: The index of the center atom.
            potential_path: Path to the current ACE potential file (.yace).

        Returns:
            Atoms: The relaxed small periodic cell.
        """
        # 1. Extraction & PBC Setup
        if atoms.pbc.any() and atoms.cell.volume > 0:
            vectors = atoms.get_distances(
                center_id, range(len(atoms)), mic=True, vector=True
            )
        else:
            vectors = atoms.positions - atoms.positions[center_id]

        half_box = self.box_size / 2.0
        mask = (np.abs(vectors) <= half_box).all(axis=1)

        subset_atoms = atoms[mask].copy()
        subset_atoms.positions = vectors[mask]
        subset_atoms.positions += half_box
        subset_atoms.set_cell([self.box_size, self.box_size, self.box_size])
        subset_atoms.set_pbc(True)
        subset_atoms.wrap()

        # 2. Stoichiometry Check
        self._check_stoichiometry(atoms, subset_atoms)

        # 3. MLIP Constrained Relaxation
        relaxed_atoms = self._relax_cell(subset_atoms, potential_path)

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
        element_map = " ".join(self.elements) if self.elements else " ".join(sorted(list(set(atoms.get_chemical_symbols()))))

        parameters = {
            "pair_style": "pace",
            "pair_coeff": [f"* * {potential_path} {element_map}"],
            "mass": ["* 1.0"],
        }

        calc = LAMMPS(
            command=self.lammps_cmd,
            parameters=parameters,
            specorder=self.elements if self.elements else None,
            files=[potential_path],
            keep_tmp_files=False,
        )

        atoms.calc = calc

        # Constraint: Fix atoms within r_core
        box_center = np.array([self.box_size / 2.0] * 3)
        rel_pos = atoms.positions - box_center
        L = self.box_size
        rel_pos = rel_pos - L * np.round(rel_pos / L)
        dists = np.linalg.norm(rel_pos, axis=1)

        fixed_indices = np.where(dists < self.r_core)[0]

        if len(fixed_indices) > 0:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
        else:
            logger.warning("No atoms found within r_core to fix.")

        try:
            ucf = ExpCellFilter(atoms)
            opt = FIRE(ucf, logfile=None)
            opt.run(fmax=0.05, steps=100)
        except Exception as e:
            logger.error(f"Relaxation failed: {e}")
            pass

        return atoms
