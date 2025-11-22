"""Active Learning components for structure generation and sampling.

This module provides functionality to generate small periodic cells around
uncertain local environments and to sample those environments.
"""

import logging
import numpy as np
from typing import List, Dict, Optional

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

        Raises:
            ValueError: If gamma values are not found in atoms.arrays.
        """
        # Try to find the gamma array
        # LAMMPS dump usually outputs 'f_f_gamma' if we used 'dump ... f_f_gamma'
        gamma_key = None
        possible_keys = ["f_f_gamma", "f_gamma", "gamma"]

        for key in possible_keys:
            if key in atoms.arrays:
                gamma_key = key
                break

        if not gamma_key:
            available_keys = list(atoms.arrays.keys())
            raise ValueError(f"Gamma values not found in atoms.arrays. Available keys: {available_keys}. "
                             "Ensure the dump file contains the gamma calculation output.")

        gammas = atoms.get_array(gamma_key)
        if gammas.ndim > 1:
            gammas = gammas.flatten()

        # Get indices of top n_clusters gammas
        # np.argsort returns ascending, so we take the last n
        sorted_indices = np.argsort(gammas)
        # Take top n_clusters, reversing to get descending order (highest gamma first)
        top_indices = sorted_indices[-n_clusters:][::-1]

        logger.info(f"Selected {len(top_indices)} atoms. Max gamma: {gammas[top_indices[0]]:.4f}")

        return [int(idx) for idx in top_indices]


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
    ):
        """Initialize the SmallCellGenerator.

        Args:
            r_core: Radius for the core region where atoms are fixed during relaxation.
            box_size: Size of the cubic small cell (Angstroms).
            stoichiometric_ratio: Expected stoichiometry.
            lammps_cmd: Command to run LAMMPS.
            min_bond_distance: Minimum bond distance for overlap removal.
            stoichiometry_tolerance: Tolerance for stoichiometry check.
        """
        self.r_core = r_core
        self.box_size = box_size
        self.stoichiometric_ratio = stoichiometric_ratio
        self.lammps_cmd = lammps_cmd
        self.min_bond_distance = min_bond_distance
        self.stoichiometry_tolerance = stoichiometry_tolerance

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

        # Ensure we handle PBC correctly when extracting
        # We want a box of size self.box_size centered at atoms[center_id]

        # Get positions relative to the center atom, accounting for PBC of the large cell
        # if the large cell has PBC.
        if large_atoms.pbc.any():
            # Use get_distances to handle MIC
            # vector=True returns the vector pointing from center to others
            vectors = large_atoms.get_distances(
                center_id, range(len(large_atoms)), mic=True, vector=True
            )
        else:
            vectors = large_atoms.positions - large_atoms.positions[center_id]

        half_box = self.box_size / 2.0

        # Select atoms within the cubic box [-half_box, +half_box] in all dimensions
        mask = (np.abs(vectors) <= half_box).all(axis=1)

        subset_atoms = large_atoms[mask].copy()

        # Update positions to be relative to the center, then center them in the new box
        # The new box is [0, box_size]^3. The center is at [box_size/2]^3.
        # subset_vectors = vectors[mask]
        # subset_atoms.positions = subset_vectors + half_box

        # Wait, get_distances(a, b) returns vector from a to b.
        # So positions relative to center are exactly `vectors`.
        subset_atoms.positions = vectors[mask] + half_box

        # Set new cell and PBC
        subset_atoms.set_cell([self.box_size, self.box_size, self.box_size])
        subset_atoms.set_pbc(True)
        subset_atoms.wrap() # Ensure all atoms are inside the box

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
            # Get distance matrix with MIC
            dists = atoms.get_all_distances(mic=True)
            # Set diagonal to infinity to ignore self-interaction
            np.fill_diagonal(dists, np.inf)

            # Find pairs with distance < min_bond_distance
            # triu to avoid duplicates
            overlap_indices = np.argwhere(np.triu(dists < self.min_bond_distance))

            if len(overlap_indices) == 0:
                break

            to_delete = set()
            for i, j in overlap_indices:
                if i in to_delete or j in to_delete:
                    continue

                # Determine which to delete: the one further from center
                # (Assuming center_pos is the center of the box where the core atom is)
                # Note: atoms.positions are already in [0, box_size]
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

            # Delete atoms (sort descending to avoid index shift issues)
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

        # Check against self.stoichiometric_ratio
        # We assume the ratio is normalized or at least relative.
        # Example: {'A': 1, 'B': 1} -> 0.5, 0.5
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
        # Determine elements for LAMMPS
        unique_elements = sorted(list(set(atoms.get_chemical_symbols())))
        element_map = " ".join(unique_elements)

        # Setup LAMMPS Calculator
        # We use 'pace' pair style
        parameters = {
            "pair_style": "pace",
            "pair_coeff": [f"* * {potential_path} {element_map}"],
            "mass": ["* 1.0"], # specific masses are usually handled by ASE if atoms object has them, but here we set dummy
        }

        # Note: ASE's LAMMPS calculator usually handles masses if they are in atoms object.
        # But we might need to ensure the types map correctly.
        # For 'pace', the element mapping string tells LAMMPS which element maps to which type.

        calc = LAMMPS(
            command=self.lammps_cmd,
            parameters=parameters,
            specorder=unique_elements, # Ensures type 1 is first element, etc.
            files=[potential_path],
            keep_tmp_files=False,
        )

        atoms.calc = calc

        # 3. Constraint: Fix atoms within r_core
        # Calculate distances from the center of the box
        box_center = np.array([self.box_size / 2.0] * 3)

        # We must respect PBC when calculating distance to center of the box
        # Since the box is cubic and atoms are wrapped, direct distance to center is fine
        # if we assume standard MIC, but let's be precise.
        # Actually, the center is fixed at box_size/2. The atoms are in [0, box_size].
        rel_pos = atoms.positions - box_center
        # Apply MIC for the small cell dimensions
        L = self.box_size
        rel_pos = rel_pos - L * np.round(rel_pos / L)
        dists = np.linalg.norm(rel_pos, axis=1)

        fixed_indices = np.where(dists < self.r_core)[0]

        if len(fixed_indices) > 0:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
            logger.debug(f"Fixed {len(fixed_indices)} atoms within r_core={self.r_core}")
        else:
            logger.warning("No atoms found within r_core to fix.")

        # 4. Relaxation
        try:
            # Use ExpCellFilter to relax cell shape and positions (of non-fixed atoms)
            # The prompt asks to relax "Cell shape and Buffer atoms".
            ucf = ExpCellFilter(atoms)
            opt = FIRE(ucf, logfile=None)
            opt.run(fmax=0.05, steps=200) # Increased steps slightly to ensure convergence
        except Exception as e:
            logger.error(f"Relaxation failed: {e}. Proceeding with unrelaxed structure.")

        return atoms
