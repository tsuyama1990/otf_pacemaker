"""Small Cell Generator Strategy."""

import logging
import numpy as np
from typing import Dict, Optional, List
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
        pass # Will raise usually if not available during init or use

from src.core.interfaces import StructureGenerator
from src.autostructure.preopt import PreOptimizer

logger = logging.getLogger(__name__)

class SmallCellGenerator(StructureGenerator):
    """Responsible for generating and relaxing small periodic cells."""

    def __init__(
        self,
        r_core: float,
        box_size: float,
        stoichiometric_ratio: Dict[str, float],
        lammps_cmd: str = "lmp_serial",
        min_bond_distance: float = 1.5,
        stoichiometry_tolerance: float = 0.1,
        lj_params: Optional[Dict[str, float]] = None,
        elements: Optional[List[str]] = None
    ):
        """Initialize the SmallCellGenerator.

        Args:
            r_core: Radius for the core region where atoms are fixed during relaxation.
            box_size: Size of the cubic small cell (Angstroms).
            stoichiometric_ratio: Expected stoichiometry.
            lammps_cmd: Command to run LAMMPS.
            min_bond_distance: Minimum bond distance for overlap removal.
            stoichiometry_tolerance: Tolerance for stoichiometry check.
            lj_params: Optional LJ params for PreOptimizer (if needed).
            elements: Optional list of elements for PreOptimizer (if needed).
        """
        self.r_core = r_core
        self.box_size = box_size
        self.stoichiometric_ratio = stoichiometric_ratio
        self.lammps_cmd = lammps_cmd
        self.min_bond_distance = min_bond_distance
        self.stoichiometry_tolerance = stoichiometry_tolerance

        # Initialize PreOptimizer if params provided (optional enhancement)
        # SmallCellGenerator mainly uses ACE potential for relaxation,
        # but could use PreOptimizer for initial cleanup if desired.
        # Currently kept for compatibility with Factory injection.
        self.pre_optimizer = None
        if lj_params:
            self.pre_optimizer = PreOptimizer(lj_params=lj_params, emt_elements=set(elements) if elements else None)

    def generate_cell(self, large_atoms: Atoms, center_id: int, potential_path: str) -> Atoms:
        """Generate and relax a small periodic cell around a center atom.

        Args:
            large_atoms: The full atomic structure.
            center_id: The index of the center atom.
            potential_path: Path to the current ACE potential file (.yace).

        Returns:
            Atoms: The relaxed small periodic cell.
        """
        # 1. Rectangle Extraction (Cubic Box)

        if large_atoms.pbc.any():
            vectors = large_atoms.get_distances(
                center_id, range(len(large_atoms)), mic=True, vector=True
            )
        else:
            vectors = large_atoms.positions - large_atoms.positions[center_id]

        half_box = self.box_size / 2.0
        mask = (np.abs(vectors) <= half_box).all(axis=1)

        subset_atoms = large_atoms[mask].copy()
        subset_atoms.positions = vectors[mask] + half_box

        subset_atoms.set_cell([self.box_size, self.box_size, self.box_size])
        subset_atoms.set_pbc(True)
        subset_atoms.wrap()

        # 1.1. Overlap Removal
        self._remove_overlaps(subset_atoms, center_pos=np.array([half_box, half_box, half_box]))

        # 1.2. Stoichiometry Check
        self._check_stoichiometry(subset_atoms)

        # 2. MLIP Constrained Relaxation
        relaxed_atoms = self._relax_cell(subset_atoms, potential_path)

        return relaxed_atoms

    def _remove_overlaps(self, atoms: Atoms, center_pos: np.ndarray):
        """Remove overlapping atoms based on min_bond_distance."""
        while True:
            dists = atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)

            overlap_indices = np.argwhere(np.triu(dists < self.min_bond_distance))

            if len(overlap_indices) == 0:
                break

            to_delete = set()
            for i, j in overlap_indices:
                if i in to_delete or j in to_delete:
                    continue

                pos_i = atoms.positions[i]
                pos_j = atoms.positions[j]

                dist_i = np.linalg.norm(pos_i - center_pos)
                dist_j = np.linalg.norm(pos_j - center_pos)

                if dist_i > dist_j:
                    to_delete.add(i)
                else:
                    to_delete.add(j)

            if not to_delete:
                 break

            del atoms[sorted(list(to_delete), reverse=True)]
            logger.info(f"Removed {len(to_delete)} overlapping atoms.")

    def _check_stoichiometry(self, atoms: Atoms):
        """Check if the stoichiometry matches the target ratio."""
        symbols = atoms.get_chemical_symbols()
        total_atoms = len(symbols)
        if total_atoms == 0:
            return

        counts = {}
        for s in symbols:
            counts[s] = counts.get(s, 0) + 1

        target_sum = sum(self.stoichiometric_ratio.values())

        for elem, target_val in self.stoichiometric_ratio.items():
            target_frac = target_val / target_sum
            actual_frac = counts.get(elem, 0) / total_atoms

            diff = abs(actual_frac - target_frac)
            if diff > self.stoichiometry_tolerance:
                logger.warning(
                    f"Stoichiometry warning for {elem}: Expected {target_frac:.2f}, Got {actual_frac:.2f} "
                    f"(Tolerance: {self.stoichiometry_tolerance})"
                )

    def _relax_cell(self, atoms: Atoms, potential_path: str) -> Atoms:
        """Relax the cell using the given potential with constraints.

        Args:
            atoms: The atoms to relax.
            potential_path: Path to the .yace potential.

        Returns:
            Atoms: The relaxed structure.
        """
        unique_elements = sorted(list(set(atoms.get_chemical_symbols())))
        element_map = " ".join(unique_elements)

        parameters = {
            "pair_style": "pace",
            "pair_coeff": [f"* * {potential_path} {element_map}"],
            "mass": ["* 1.0"],
        }

        calc = LAMMPS(
            command=self.lammps_cmd,
            parameters=parameters,
            specorder=unique_elements,
            files=[potential_path],
            keep_tmp_files=False,
        )

        atoms.calc = calc

        box_center = np.array([self.box_size / 2.0] * 3)
        rel_pos = atoms.positions - box_center
        L = self.box_size
        rel_pos = rel_pos - L * np.round(rel_pos / L)
        dists = np.linalg.norm(rel_pos, axis=1)

        fixed_indices = np.where(dists < self.r_core)[0]

        if len(fixed_indices) > 0:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
            logger.debug(f"Fixed {len(fixed_indices)} atoms within r_core={self.r_core}")
        else:
            logger.warning("No atoms found within r_core to fix.")

        try:
            ucf = ExpCellFilter(atoms)
            opt = FIRE(ucf, logfile=None)
            opt.run(fmax=0.05, steps=200)
        except Exception as e:
            logger.error(f"Relaxation failed: {e}. Proceeding with unrelaxed structure.")

        return atoms
